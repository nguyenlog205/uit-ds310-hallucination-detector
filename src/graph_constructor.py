import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import pandas as pd
import networkx as nx
from tqdm import tqdm
from utils import load_configs
import re
import matplotlib.pyplot as plt
import json

from underthesea import ner

class GraphConstructor:
    RE_PROMPT = (
        "You are an expert Knowledge Graph Engineer. Your task is to extract entities and relations from Vietnamese text into JSON format.\n\n"
        "### STRICT LOGIC RULES (Phi-3.5 Logic Engine):\n"
        "1. **Subject-Object Causality**: Always identify WHO did WHAT to WHOM.\n"
        "   - WRONG: [Company] --founded--> [Person]\n"
        "   - RIGHT: [Person] --founded--> [Company]\n"
        "2. **Coreference Resolution**: Replace pronouns ('Ông ấy', 'Nam ca sĩ') with the specific Proper Name found in text.\n"
        "3. **Entity Merging**: Use the real name as 'head'. Put aliases/titles in 'head_props'.\n\n"
        "### JSON SCHEMA:\n"
        "[\n"
        "  {\n"
        "    \"head\": \"Entity1\", \"head_props\": {\"key\": \"value\"},\n"
        "    \"relation\": \"action\", \"relation_props\": {\"time\": \"...\", \"loc\": \"...\"},\n"
        "    \"tail\": \"Entity2\", \"tail_props\": {...}\n"
        "  }\n"
        "]\n\n"
        "### EXAMPLES:\n"
        "Input: \"Ông Vượng lập VinGroup năm 2000.\"\n"
        "Output:\n"
        "[{\"head\": \"Phạm Nhật Vượng\", \"head_props\": {\"chuc_vu\": \"Ông\"}, \"relation\": \"thành lập\", \"relation_props\": {\"thoi_gian\": \"2000\"}, \"tail\": \"VinGroup\", \"tail_props\": {}}]\n\n"
        "Input: \"Sơn Tùng ra mắt Lạc Trôi.\"\n"
        "Output:\n"
        "[{\"head\": \"Nguyễn Thanh Tùng\", \"head_props\": {\"nghe_danh\": \"Sơn Tùng M-TP\"}, \"relation\": \"ra mắt\", \"relation_props\": {}, \"tail\": \"Lạc Trôi\", \"tail_props\": {\"loai\": \"Bài hát\"}}]\n\n"
        "### RESPONSE INSTRUCTION:\n"
        "Analyze the text logic carefully. Output **ONLY** the valid JSON list. Do not explain."
    )

    def __init__(self, config_path: str):
        print(">> Loading Configuration...")
        config = load_configs(config_path)
        gc_config = config.get('GraphConstructor', {})
        
        # --- Load Model RE ---
        self.re_model_name = gc_config.get('RE_MODEL_NAME', "unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
        print(f">> Loading RE Model: {self.re_model_name}...")
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.re_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.re_model = AutoModelForCausalLM.from_pretrained(
            self.re_model_name,
            device_map='cuda:0',
            dtype=dtype,
            low_cpu_mem_usage=True
        ).eval()

        # --- Load Model NER ---
        print(">> NER Model: Underthesea (Ready)") 

        self.graph = nx.DiGraph()
        self.config = gc_config

    # =======================================
    # Internal Methods
    # =======================================
    # --- Extract Entities ---
    def _extract_entities(self, text: str) -> List[str]:
        """
        BƯỚC 1: NER với thuật toán 'Nối Danh Từ' (Chunking)
        Khắc phục lỗi tách từ rời rạc của Underthesea.
        """
        try:
            raw_ner = ner(text)
        except Exception as e:
            print(f"[NER Error] {e}")
            return []
        
        entities = []
        current_chunk = []

        VALID_TAGS = ['Np', 'N', 'Nu', 'Ny', 'M', 'P'] 
        
        for item in raw_ner:
            word = item[0]
            tag = item[1]
            
            if word and word[0].isupper() and word.isalnum():
                current_chunk.append(word)
            elif tag in VALID_TAGS:
                current_chunk.append(word)
            else:
                if current_chunk:
                    self._add_chunk_to_entities(current_chunk, entities)
                    current_chunk = []
        if current_chunk:
            self._add_chunk_to_entities(current_chunk, entities)
        
        return list(set(entities))

    def _add_chunk_to_entities(self, chunk, entities_list):
        full_entity = " ".join(chunk)
        if len(full_entity) > 1 or (len(full_entity) == 1 and full_entity.isupper()):
            entities_list.append(full_entity)

    def _clean_stutter(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) > 1 and len(tokens[0]) < len(tokens[1]) and tokens[1].startswith(tokens[0]):
            return " ".join(tokens[1:])
        return text

    # --- Extract Relations ---
    def _extract_relations(self, text: str, entities: List[str]) -> List[dict]:
        if len(entities) < 2:
            return []
            
        entity_str = ", ".join(f"'{e}'" for e in entities)
        
        user_content = (
            f"Văn bản: \"{text}\"\n"
            f"Thực thể GỢI Ý (từ NER): [{entity_str}]\n"
            "CHÚ Ý: Danh sách trên có thể thiếu. Hãy tự tìm thêm các thực thể khác trong văn bản (như tên người nước ngoài, tên bài hát, địa danh) để tạo quan hệ đầy đủ nhất."
        )

        messages = [
            {"role": "system", "content": self.RE_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.re_model.device)
        
        with torch.no_grad():
            outputs = self.re_model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.1
            )
            
        decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        del inputs, outputs
        torch.cuda.empty_cache()

        try:
            json_str = decoded.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            results = json.loads(json_str)
            final_results = []
            if isinstance(results, list):
                for item in results:
                    if "head" in item and "tail" in item and "relation" in item:
                        item["head"] = self._clean_stutter(item["head"])
                        item["tail"] = self._clean_stutter(item["tail"])
                        final_results.append(item)
            
            return final_results

        except json.JSONDecodeError:
            print(f"[JSON Error] Model trả về format sai:\n{decoded[:100]}...")
            return []
        except Exception as e:
            print(f"[Error] {e}")
            return []

    def _update_graph(self, relations: List[dict]):
        for item in relations:
            head = item.get("head")
            tail = item.get("tail")
            rel_label = item.get("relation")
            head_props = item.get("head_props", {})
            tail_props = item.get("tail_props", {})
            rel_props = item.get("relation_props", {})
            self.graph.add_node(head, **head_props)
            self.graph.add_node(tail, **tail_props)
            
            edge_attrs = {"relation": rel_label}
            edge_attrs.update(rel_props) 
            
            self.graph.add_edge(head, tail, **edge_attrs)

    # =======================================
    # Public Methods
    # =======================================
    def process(self, dataset: pd.DataFrame, text_column: str):
        data = dataset[text_column].tolist()
        
        print(f"Processing {len(data)} items...")
        for text in tqdm(data, desc="Building Graph"):
            # --- NER ---
            entities = self._extract_entities(text)
            if entities:
                triples = self._extract_relations(text, entities)
                self._update_graph(triples)
                
        print(f"Done. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        return self.graph
    

def example():
    data = {
        'content': [
            "Nguyễn Thanh Tùng, nghệ danh Sơn Tùng M-TP, sinh năm 1994 tại Thái Bình. Anh thành lập công ty M-TP Entertainment vào năm 2016 và hiện giữ chức chủ tịch. Năm 2019, nam ca sĩ hợp tác với huyền thoại Snoop Dogg để phát hành bom tấn Hãy Trao Cho Anh tại Mỹ",
            "Phạm Nhật Vượng sinh năm 1968 tại Hà Nội. Ông thành lập công ty Technocom tại Ukraine vào năm 1993 trước khi về Việt Nam xây dựng Vingroup. Năm 2017, Vingroup chính thức ra mắt thương hiệu ô tô VinFast và đặt tổ hợp nhà máy sản xuất hiện đại tại Hải Phòng."
        ]
    }
    df = pd.DataFrame(data)
    print("=== Dữ liệu đầu vào ===")
    print(df)
    print("="*30)

    # --- Run ---
    gc = GraphConstructor(config_path=r'configs\graph_constructor.yml')
    kg = gc.process(dataset=df, text_column='content')

    # --- Result Check ---
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ PROPERTY GRAPH ({kg.number_of_nodes()} Nodes, {kg.number_of_edges()} Edges)")
    print("="*50)
    
    for u, v, data in kg.edges(data=True):
        rel = data.get('relation')

        edge_props = {k: v for k, v in data.items() if k != 'relation'}
        edge_props_str = str(edge_props) if edge_props else ""
        
        u_props = kg.nodes[u]
        u_props_str = str(u_props) if u_props else ""
        
        print(f"🔹 [{u}] --{rel}--> [{v}]")
        if u_props_str:    print(f"    ├─ Head Props: {u_props_str}")
        if edge_props_str: print(f"    ├─ Rel Props:  {edge_props_str}")
        print("")

    # --- Visualize ---
    if kg.number_of_edges() > 0:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(kg, k=0.5)
        
        nx.draw_networkx_nodes(kg, pos, node_size=2000, node_color='lightblue', alpha=0.9)
        nx.draw_networkx_labels(kg, pos, font_size=10, font_family='sans-serif')
        
        nx.draw_networkx_edges(kg, pos, width=2, alpha=0.6, edge_color='gray', arrows=True)
        
        edge_labels = nx.get_edge_attributes(kg, 'relation')
        nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_size=9, font_color='red')
        
        plt.title("Knowledge Graph Demo")
        plt.axis('off')
        plt.show()
    else:
        print("Không tìm thấy quan hệ nào để vẽ.")

if __name__ == "__main__":
    example()