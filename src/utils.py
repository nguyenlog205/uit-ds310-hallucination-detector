import yaml

def load_configs(config_path: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file config tại '{config_path}'")
        return {}
    except yaml.YAMLError as exc:
        print(f"Lỗi cú pháp trong file YAML: {exc}")
        return {}
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return {}

import json
from networkx.readwrite import json_graph

class KGReader:
    def __init__(self):
        pass

    def save_to_json(self,graph, filename="knowledge_graph.json"):

        data = json_graph.node_link_data(graph)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ Đã lưu graph vào {filename}")
    
    def load_from_json(filename="knowledge_graph.json"):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Khôi phục lại thành NetworkX Graph
        graph = json_graph.node_link_graph(data)
        print(f"✅ Đã load graph: {graph.number_of_nodes()} nodes")
        return graph