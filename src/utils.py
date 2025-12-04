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