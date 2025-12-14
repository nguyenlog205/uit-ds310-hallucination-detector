import torch
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from utils import load_configs
from typing import Tuple, List
import unicodedata
import pandas as pd
from tqdm import tqdm

class PromptPreprocessor:
    # =========================================================
    # CONFIGURATION AND DECLARATION
    # =========================================================
    SYSTEM_PROMPT = (
        "Bạn là một trợ lý AI chuyên sửa lỗi chính tả tiếng Việt.\n"
        "Nhiệm vụ: Chuyển đổi câu sai chính tả thành câu đúng chuẩn ngữ pháp và dấu câu.\n"
        "Yêu cầu tuyệt đối:\n"
        "1. Giữ nguyên ý nghĩa và số lượng từ của câu gốc.\n"
        "2. Chỉ sửa lỗi chính tả, teencode (vd: 'k' -> 'không', 'j' -> 'gì').\n"
        "3. KHÔNG thêm bớt từ, KHÔNG giải thích, chỉ trả về câu đã sửa."
    )

    def __init__(self, config_path: str):
        # --- Load configurations ---
        config = load_configs(config_path)
        pp_config = config['PromptPreprocessor'] 
        self.MODEL_NAME = pp_config['MODEL_NAME']
        self.MAX_TOKENS = pp_config['MAX_TOKENS']
        
        self.MAX_INPUT_LENGTH = pp_config.get('MAX_INPUT_LENGTH', 1024)

        self.TEMPERATURE = pp_config['TEMPERATURE']
        self.TOP_P = pp_config['TOP_P']
        self.BATCH_SIZE = pp_config.get('BATCH_SIZE', 4)
        
        # --- Load model and tokenizer ---
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            use_fast=True,
            padding_side='left' 
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.pad_id = self.tokenizer.pad_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            device_map='cuda:0', 
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()

    # =========================================================
    # CORE LOGIC
    # =========================================================
    
    def _is_semantically_preserved(self, inp: str, out: str) -> bool:
        """
        Bộ lọc phiên bản mới: Chỉ kiểm tra độ lệch số lượng từ.
        Cho phép sai số +/- 20% số từ (để xử lý trường hợp tách/gộp từ).
        """
        inp_words = inp.strip().split()
        out_words = out.strip().split()
        
        len_in = len(inp_words)
        len_out = len(out_words)
        
        if len_in == 0: return False
        if abs(len_in - len_out) > 2:
            return False
            
        return True

    def _build_chat_messages(self, input_text: str):
        """Tạo cấu trúc Chat chuẩn cho Qwen"""
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            # Few-shot example 1
            {"role": "user", "content": "ngay hum nay troi dep qua minh di choi nhe"},
            {"role": "assistant", "content": "ngày hôm nay trời đẹp quá mình đi chơi nhé"},
            # Few-shot example 2
            {"role": "user", "content": "Th3 nao la cay an qa"},
            {"role": "assistant", "content": "Thế nào là cây ăn quả"},
            # Input thực tế
            {"role": "user", "content": input_text}
        ]

    def _generate_batch(self, raw_texts: List[str]) -> List[str]:
        if not raw_texts: return []
        prompts = []
        for text in raw_texts:
            messages = self._build_chat_messages(text)
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(prompt_str)

        enc = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=self.MAX_INPUT_LENGTH
        ).to(self.model.device)
        
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                top_p=self.TOP_P,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.pad_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = out[:, input_len:]
        decoded_list = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        del enc, out, generated_ids
        torch.cuda.empty_cache()
        results = []
        for txt in decoded_list:
            results.append(txt.strip())
            
        return results

    def _correct_batch(self, raw_texts: List[str]) -> List[str]:
        outputs = self._generate_batch(raw_texts)
        
        final_results = []
        for inp, out in zip(raw_texts, outputs):
            if out and self._is_semantically_preserved(inp, out):
                final_results.append(out)
            else:
                final_results.append(inp) 
        return final_results

    # ================================================================
    # CORE BUILD-IN METHODS
    # ================================================================
    # --- Run an example ---
    def run_example(
        self,
        sample_prompt=[
            "Th3 nao la cay an qa",
            "ngay hum nay troi dep qua minh di choi nhe",
            "ê pà owy cái này dùq xao dọ"
        ]
    ):
        import time

        print(f'=== CONFIG ===\nModel: {self.MODEL_NAME}\nBatch Size: {self.BATCH_SIZE}')
        starting = time.time()
        outputs = self._correct_batch(sample_prompt)
        ending = time.time()
        duration = ending - starting

        for i, (inp, out) in enumerate(zip(sample_prompt, outputs), 1):
            print(f"{i}. IN : {inp}")
            print(f"   OUT: {out}")
            print("-" * 20)

        print(f'Total Time: {duration:.2f}s')
        print(f'Speed: {len(sample_prompt) / duration:.2f} sentences/sec')
        
        return None

    def process(self, dataset: pd.DataFrame, input_column: str, output_column: str, batch_size: int = None):
        bs = batch_size if batch_size else self.BATCH_SIZE
        
        df = dataset.copy()
        if input_column not in dataset.columns:
            raise ValueError(f"Input column '{input_column}' not found.")
        
        df.dropna(subset=[input_column], inplace=True)
        df[input_column] = df[input_column].astype(str)
        
        all_data = df[input_column].tolist()
        all_corrected_text = []

        print(f"Processing {len(df)} rows with batch_size={bs}...")
        for i in tqdm(range(0, len(all_data), bs), desc="Batch Processing"):
            batch_texts = all_data[i : i + bs]
            try:
                batch_results = self._correct_batch(batch_texts)
                all_corrected_text.extend(batch_results)
            except Exception as e:
                print(f"[Error] Batch {i} failed: {e}")
                all_corrected_text.extend(batch_texts)

        if len(all_corrected_text) == len(df):
            df[output_column] = all_corrected_text
        return df
    
prompt_preprocessor = PromptPreprocessor(
    config_path=r'configs\prompt_preprocessor.yml'
)
prompt_preprocessor.run_example()