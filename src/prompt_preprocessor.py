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
    STRICT_RULE = (
        "You are a tool that ONLY corrects Vietnamese spelling mistakes.\n"
        "Your task is to perform a strict 1-to-1 mapping from misspelled words to their correct versions.\n"
        '- Fix misspellings, add correct diacritics, and normalize characters (e.g., "dd" -> "đ", "0/6" -> "o/ô", "3" -> "e/ê").\n'
        "- DO NOT add, remove, or change any words.\n"
        "- DO NOT change the meaning.\n"
        "- KEEP the same number of words and their order as the input.\n"
        "- Output ONLY the corrected sentence (no explanation, no symbols, no quotes, no <think>)."
    )

    FEW_SHOT = textwrap.dedent("""
        Examples:
        Input: "ngay hum nay troi dep qua minh di choi nhe"
        Output: "ngày hôm nay trời đẹp quá mình đi chơi nhé"

        Input: "neus khong phai ngời ban đia thi ngwoi lao dong se bi xu phat nhu th3 nao?"
        Output: "nếu không phải người bản địa thì người lao động sẽ bị xử phạt như thế nào?"

        Input: "toi ddang ddi dou tke ve viec lam"
        Output: "tôi đang đi đâu kể về việc làm"
    """).strip()

    ULTRA_NOTE = (
        "Additional constraints:\n"
        "- Do NOT add auxiliary words like 'sẽ/will'.\n"
        "- Do NOT flip meanings (e.g., 'bình thường' != 'bất thường')."
    )

    EXTRA_EXAMPLE = textwrap.dedent("""
        Input: "trg nhg ngay th3 troi mưa to, toi van ddi hc bt thuong"
        Output: "trong những ngày thế trời mưa to, tôi vẫn đi học bình thường"
    """).strip()

    def __init__(self, config_path: str):
        # --- Load configurations ---
        config = load_configs(config_path)
        pp_config = config['PromptPreprocessor'] 
        self.MODEL_NAME = pp_config['MODEL_NAME']
        self.MAX_TOKENS = pp_config['MAX_TOKENS']
        self.TEMPERATURE = pp_config['TEMPERATURE']
        self.TOP_P = pp_config['TOP_P']
        self.BATCH_SIZE = pp_config.get('BATCH_SIZE', 1)
        
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
            device_map='auto',
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()

    # =========================================================
    # Internally-using methods
    # =========================================================
    # --- Semantics Validation ---
    @staticmethod
    def _strip_accents(s: str) -> str:
        """Loại bỏ dấu câu (Static vì không cần self)"""
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", s)

    @staticmethod
    def _precanonical_char_subs(s: str) -> str:
        """Chuẩn hoá teencode cơ bản"""
        s = re.sub(r"(?i)\bdd\b", "đ", s)
        s = s.replace("0", "o").replace("6", "o")
        s = s.replace("3", "e")
        return s

    def _is_semantically_preserved(self, inp: str, out: str) -> bool:
        def get_base(text):
            toks = text.strip().split()
            return [self._precanonical_char_subs(self._strip_accents(t.lower())) for t in toks]

        in_base = get_base(inp)
        out_base = get_base(out)

        if len(in_base) != len(out_base):
            return False
            
        return all(a == b for a, b in zip(in_base, out_base))
    
    # --- Build Prompt ---
    def _build_strict_prompt(self, input_text: str):
        return (
            f"{self.STRICT_RULE}\n\n"
            f"{self.FEW_SHOT}\n\n"
            f"Input:\n{input_text.strip()}\n\n"
            f"Output:"
        )
    def _build_ultra_prompt(self, input_text: str):
        return (
            f"{self.STRICT_RULE}\n\n"
            f"{self.ULTRA_NOTE}\n"
            f"{self.FEW_SHOT}\n\n"
            f"{self.EXTRA_EXAMPLE}\n\n"
            f"Input:\n{input_text.strip()}\n\n"
            f"Output:"
        )
    
    # --- Generation and correction ---
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []

        enc = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=2048
        )
        
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                top_p=self.TOP_P,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.pad_id
            )
        
        input_len = enc["input_ids"].shape[1]
        generated_tokens = out[:, input_len:]
        
        decoded_list = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        results = []
        for txt in decoded_list:
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
            txt = txt.replace("```", "").strip()
            if "Output:" in txt:
                txt = txt.split("Output:", 1)[-1].strip()
            
            lines = [ln.strip("‘’“”\"'` ").strip() for ln in txt.splitlines() if ln.strip()]
            results.append(lines[0] if lines else "")
        
        del enc
        del out
        del generated_tokens
        torch.cuda.empty_cache()
        
        return results

    def _correct_batch(self, raw_texts: List[str]) -> List[str]:
        """Wrapper để xử lý logic kiểm tra ngữ nghĩa cho cả batch"""
        prompts = [self._build_ultra_prompt(t) for t in raw_texts]
        outputs = self._generate_batch(prompts)
        final_results = []
        for inp, out in zip(raw_texts, outputs):
            if out and self._is_semantically_preserved(inp, out):
                final_results.append(out)
            else:
                final_results.append(inp) 
        return final_results


    # =========================================================
    # Build-in methods 
    # =========================================================
    # --- Run an example ---
    def run_example(
        self,
        sample_prompt = [
            " Th3 nao la cay an qa, va cay an qua co nguon goc tu dau"
        ]
    ) -> None:
        print(f'=== SELF TEXT ===')
        print('PromptPreprocessor Configuration\n',
              'model_name:', self.MODEL_NAME,
              'max_tokens:', self.MAX_TOKENS,
              'temperature:', self.TEMPERATURE,
              'top_p:', self.TOP_P,
              'batch_size:', self.BATCH_SIZE,
              sep='\n'
        )
        outputs = self._correct_batch(sample_prompt)
        
        for i, (inp, out) in enumerate(zip(sample_prompt, outputs), 1):
            print(f"{i}. IN : {inp}")
            print(f"   OUT: {out}")
            print("-" * 20)
        print("=================")
        return None
    
    # --- MAIN BUILD-IN METHOD ---
    def process(
        self,
        dataset: pd.DataFrame,
        input_column: str,
        output_column: str,
        batch_size: int = 8
    ):
        df = dataset.copy()
        if input_column not in dataset.columns:
            raise ValueError(f"Input column '{input_column}' not found.")
        
        df.dropna(subset=[input_column], inplace=True)
        df[input_column] = df[input_column].astype(str)
        
        all_data = df[input_column].tolist()
        all_corrected_text = []

        print(f"Processing {len(df)} rows with batch_size={batch_size}...")
        for i in tqdm(range(0, len(all_data), batch_size), desc="Batch Processing"):
            batch_texts = all_data[i : i + batch_size]
            
            try:
                batch_results = self._correct_batch(batch_texts)
                all_corrected_text.extend(batch_results)
            except Exception as e:
                print(f"[Error] Batch {i} failed: {e}")
                all_corrected_text.extend(batch_texts)

        if len(all_corrected_text) == len(df):
            df[output_column] = all_corrected_text
        else:
            print(f"Length mismatch: Input {len(df)} vs Output {len(all_corrected_text)}")
        return df