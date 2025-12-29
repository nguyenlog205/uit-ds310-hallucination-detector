import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from tqdm import tqdm
from groq import Groq
import yaml
from networkx.readwrite import json_graph

def load_configs(path):
    print(f">> Reading config from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        if not config['GraphConstructor'].get('GROQ_API_KEY'):
            config['GraphConstructor']['GROQ_API_KEY'] = os.environ.get("GROQ_API_KEY")
            
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Không tìm thấy file config tại: {path}")
    except Exception as e:
        raise ValueError(f"❌ Lỗi đọc file YAML: {e}")

class GraphConstructor:
    E2E_PROMPT = (
        "You are an expert Vietnamese Knowledge Graph Engineer.\n"
        "Your task: Extract semantic triples (Subject, Relation, Object) with rich attributes.\n\n"
        "### RULES:\n"
        "1. **Coreference**: Replace pronouns ('Nó', 'Ông ấy') with the Real Name.\n"
        "2. **Semantic Extraction**: Capture the CORE MEANING, not just grammar.\n"
        "3. **Atomic Entities & Attributes Preservation** (VERY IMPORTANT):\n"
        "   - Keep `head` and `tail` concise (1-5 words).\n"
        "   - **MIGRATE** adjectives, definitions, or descriptive phrases into `head_props` or `tail_props`.\n"
        "   - Example: \"biến thể phản động nhất của chủ nghĩa đế quốc là phát xít\" \n"
        "     -> Tail: \"Chủ nghĩa phát xít\"\n"
        "     -> Tail_Props: {\"description\": \"biến thể phản động nhất của chủ nghĩa đế quốc\"}\n"
        "4. **Relation Context**: Extract time, location, and adverbial phrases into `relation_props`.\n\n"
        "### JSON OUTPUT:\n"
        "You must output a JSON Object with key 'graph':\n"
        "{\n"
        "  \"graph\": [\n"
        "    {\n"
        "      \"head\": \"Entity Name\",\n"
        "      \"head_props\": {\"type\": \"Person/Org/Event\", \"alias\": \"...\", \"description\": \"...\"},\n"
        "      \"relation\": \"verb\",\n"
        "      \"relation_props\": {\"time\": \"...\", \"loc\": \"...\", \"context\": \"...\"},\n"
        "      \"tail\": \"Entity Name\",\n"
        "      \"tail_props\": {\"type\": \"...\", \"description\": \"...\"}\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    def __init__(self, config_path: str):
        print(">> Loading Configuration...")
        full_config = load_configs(config_path)
        self.config = full_config.get('GraphConstructor', {})
        
        api_key = self.config.get('GROQ_API_KEY')
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY")
            
        self.client = Groq(api_key=api_key)
        self.re_model_name = self.config.get('RE_MODEL_NAME', "llama-3.3-70b-versatile")
        self.temperature = self.config.get('TEMPERATURE', 0.0)
        
        print(f">> Model: {self.re_model_name}") 
        self.graph = nx.DiGraph()

    def _clean_text(self, text):
        return text.strip().replace('"', "'")

    def _extract_e2e(self, text: str) -> List[dict]:
        messages = [
            {"role": "system", "content": self.E2E_PROMPT},
            {"role": "user", "content": f"Analyze this Vietnamese text:\n\"{text}\""}
        ]
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.re_model_name,
                temperature=self.temperature,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            
            # --- DEBUG: Bỏ comment dòng này nếu muốn xem model trả về gì ---
            data = json.loads(response_content)
            
            # Chỉ lấy dữ liệu trong key "graph"
            if "graph" in data and isinstance(data["graph"], list):
                return data["graph"]
            for key, value in data.items():
                if isinstance(value, list):
                    return value
            
            return []

        except Exception as e:
            print(f"   ❌ [API Error] {e}")
            return []

    def _update_graph(self, relations: List[dict]):
        for item in relations:
            head = item.get("head")
            tail = item.get("tail")
            rel_label = item.get("relation")
            
            if not head or not tail or not rel_label:
                continue
                
            head = self._clean_text(head)
            tail = self._clean_text(tail)
            head_props = item.get("head_props", {})
            tail_props = item.get("tail_props", {})
            rel_props = item.get("relation_props", {})
            
            self.graph.add_node(head, **head_props)
            self.graph.add_node(tail, **tail_props)
            
            edge_attrs = {"relation": rel_label}
            edge_attrs.update(rel_props)
            self.graph.add_edge(head, tail, **edge_attrs)

    def _build_graph_from_triples(self, relations: List[dict]) -> nx.DiGraph:
        G = nx.DiGraph() # Luôn tạo graph mới tinh
        
        if not relations:
            return G

        for item in relations:
            head = item.get("head")
            tail = item.get("tail")
            rel_label = item.get("relation")
            
            if not head or not tail or not rel_label:
                continue
                
            head = self._clean_text(head)
            tail = self._clean_text(tail)
            
            # Lấy thuộc tính
            head_props = item.get("head_props", {})
            tail_props = item.get("tail_props", {})
            rel_props = item.get("relation_props", {})
            
            # Add Nodes & Edge
            G.add_node(head, **head_props)
            G.add_node(tail, **tail_props)
            
            edge_attrs = {"relation": rel_label}
            edge_attrs.update(rel_props)
            G.add_edge(head, tail, **edge_attrs)
            
        return G

    def process(
        self, 
        data_input: Union[pd.DataFrame, pd.Series, List[str]], 
        text_column: str = None, 
        verbose: bool = True
    ) -> Tuple[List[nx.DiGraph], List[dict]]:
        """
        Output: (List[nx.DiGraph], List[dict])
        """
        # --- Xử lý input ---
        if isinstance(data_input, pd.DataFrame):
            data = data_input[text_column].tolist()
        elif isinstance(data_input, pd.Series):
            data = data_input.tolist()
        elif isinstance(data_input, list):
            data = data_input
        else:
            data = []

        list_of_graphs = []
        extraction_history = []

        for text in tqdm(data, desc="Building Independent Graphs", disable=not verbose):
            triples = self._extract_e2e(str(text) if text else "")
            local_graph = self._build_graph_from_triples(triples)
            list_of_graphs.append(local_graph)
            
            extraction_history.append({
                "input_text": text,
                "triples": triples
            })
        return list_of_graphs, extraction_history

def main():
    import pandas as pd
    
    # Load data
    print(">> Loading datasets...")
    train = pd.read_csv("data/origin/train.csv")
    dev = pd.read_csv("data/origin/dev.csv")
    test = pd.read_csv("data/origin/test.csv")
    
    # Gom vào list kèm tên để dễ xử lý lưu file
    datasets = [ 
        ("dev", dev), 
        ("test", test),
        ("train", train),
    ]

    kg_processor = GraphConstructor(config_path='configs/graph_constructor.yml')

    # --- Xử lý từng dataset ---
    for name, data in datasets:
        print('=' * 50)
        print(f"Processing dataset: [{name}]")
        print(f'Shape: {data.shape}')
        print('=' * 50)
        
        result = {
            'id': [],
            'context': [],
            'prompt': [],
            'response': [],
            'label': [],
        }
        
        # Dùng tqdm bọc ngoài iterrows để xem tiến độ tổng thể
        # Lưu ý: total=6 vì bạn đang break ở 5 (Test mode). 
        # Nếu chạy thật hãy bỏ 'total' hoặc để len(data) và bỏ dòng break
        for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Run {name}"):
            
            # 1. Lấy dữ liệu thô
            temp_context = row['context']
            temp_prompt = row['prompt']
            temp_response = row['response']
            
            # 2. Xử lý Graph (Batch 3 đoạn văn 1 lúc)
            # 🟢 QUAN TRỌNG: Thêm verbose=False để tắt thanh tiến trình con
            list_graphs, logs = kg_processor.process(
                [temp_context, temp_prompt, temp_response], 
                verbose=False
            )

            # 3. Trích xuất kết quả
            g_context  = list_graphs[0]
            g_prompt   = list_graphs[1]
            g_response = list_graphs[2]

            # 4. Append vào Result
            result['id'].append(row['id'])
            result['label'].append(row['label'])
            
            result['context'].append(json_graph.node_link_data(g_context))
            result['prompt'].append(json_graph.node_link_data(g_prompt))
            result['response'].append(json_graph.node_link_data(g_response))

        # 5. Lưu file (Nằm ngoài vòng lặp row, trong vòng lặp dataset)
        output_path = f'data/processed/graph_{name}.csv'
        print(f'\n>> Saving {name} graph data to {output_path}...')
        
        result_df = pd.DataFrame(result)
        result_df.to_csv(output_path, index=False)
        print("Done.")

    return

if __name__ == "__main__":
    main()