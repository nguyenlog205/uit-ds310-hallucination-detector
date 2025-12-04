import json
from pyvis.network import Network

# Dữ liệu JSON của bạn (có thể đọc từ file hoặc dán trực tiếp)
json_data = {
  "nodes": [
    {
      "id": "person_1",
      "label": "Person",
      "name": "Elena Petrova",
      "title": "Tiến sĩ",
      "attributes": {
        "former_position": "nghiên cứu viên"
      }
    },
    {
      "id": "org_1",
      "label": "Organization",
      "name": "Viện Công nghệ Sinh học Zurich",
      "acronym": "ZIB",
      "type": "viện nghiên cứu"
    },
    {
      "id": "theory_1",
      "label": "Theory",
      "name": "Lý thuyết biến đổi gene cá ngựa vằn chống virus Zika",
      "attributes": {
        "year": 2018,
        "nature": "gây tranh cãi"
      }
    },
    {
      "id": "org_2",
      "label": "Organization",
      "name": "SynBioTech",
      "type": "công ty khởi nghiệp"
    },
    {
      "id": "person_2",
      "label": "Person",
      "name": "Arthur Clarke",
      "title": "Giáo sư"
    },
    {
      "id": "org_3",
      "label": "Organization",
      "name": "Đại học Cambridge",
      "type": "trường đại học"
    },
    {
      "id": "event_1",
      "label": "LegalAction",
      "name": "Vụ kiện của ZIB chống lại SynBioTech",
      "attributes": {
        "status": "đang đối mặt"
      }
    },
    {
      "id": "tech_1",
      "label": "Technology",
      "name": "CRISPR-Cas9"
    },
    {
      "id": "asset_1",
      "label": "Asset",
      "name": "Bằng sáng chế công nghệ CRISPR-Cas9"
    },
    {
      "id": "org_4",
      "label": "Organization",
      "name": "Phòng thí nghiệm quân sự Arstotzka",
      "type": "phòng thí nghiệm quân sự"
    },
    {
      "id": "location_1",
      "label": "Location",
      "name": "Cộng hòa Arstotzka",
      "type": "quốc gia"
    },
    {
      "id": "concept_1",
      "label": "Concept",
      "name": "Lệnh cấm vận công nghệ sinh học"
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "person_1",
      "target": "org_1",
      "type": "WORKED_AT",
      "attributes": {
        "status": "cựu",
        "position": "nghiên cứu viên"
      }
    },
    {
      "id": "edge_2",
      "source": "person_1",
      "target": "theory_1",
      "type": "PUBLISHED"
    },
    {
      "id": "edge_3",
      "source": "person_1",
      "target": "org_2",
      "type": "FOUNDED"
    },
    {
      "id": "edge_4",
      "source": "person_2",
      "target": "org_2",
      "type": "CO_FOUNDED"
    },
    {
      "id": "edge_5",
      "source": "person_2",
      "target": "org_3",
      "type": "AFFILIATED_WITH",
      "attributes": {
        "affiliation": "Đại học Cambridge"
      }
    },
    {
      "id": "edge_6",
      "source": "org_1",
      "target": "event_1",
      "type": "IS_PLAINTIFF_IN"
    },
    {
      "id": "edge_7",
      "source": "org_2",
      "target": "event_1",
      "type": "IS_DEFENDANT_IN"
    },
    {
      "id": "edge_8",
      "source": "event_1",
      "target": "asset_1",
      "type": "INVOLVES_CLAIM_ABOUT",
      "attributes": {
        "claim": "sử dụng trái phép"
      }
    },
    {
      "id": "edge_9",
      "source": "org_1",
      "target": "asset_1",
      "type": "OWNS",
      "attributes": {
        "development": "phát triển từ trước đó"
      }
    },
    {
      "id": "edge_10",
      "source": "asset_1",
      "target": "tech_1",
      "type": "RELATES_TO_TECHNOLOGY"
    },
    {
      "id": "edge_11",
      "source": "event_1",
      "target": "org_4",
      "type": "INVOLVES_CLAIM_ABOUT",
      "attributes": {
        "claim": "hợp tác ngầm với"
      }
    },
    {
      "id": "edge_12",
      "source": "org_2",
      "target": "org_4",
      "type": "ALLEGED_TO_COLLABORATE_WITH",
      "attributes": {
        "manner": "ngầm"
      }
    },
    {
      "id": "edge_13",
      "source": "org_4",
      "target": "location_1",
      "type": "LOCATED_IN"
    },
    {
      "id": "edge_14",
      "source": "location_1",
      "target": "concept_1",
      "type": "UNDER_SANCTION_OF"
    }
  ]
}


def visualize_knowledge_graph(data: dict, output_filename: str = "knowledge_graph.html"):
    """
    Tạo một file HTML tương tác để hiển thị đồ thị tri thức từ dữ liệu JSON.
    (Phiên bản này có nhãn cạnh rõ ràng hơn).
    """
    # 1. Khởi tạo đồ thị
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False, directed=True)

    # 2. Tùy chỉnh vật lý và giao diện
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 18
        }
      },
      "edges": {
        "font": {
          "size": 16,
          "align": "middle",
          "color": "#ffffff",
          "strokeWidth": 5, 
          "strokeColor": "#222222"
        },
        "smooth": {
          "type": "dynamic"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 150
        },
        "minVelocity": 0.75,
        "solver": "barnesHut"
      }
    }
    """)

    # 3. Thêm các nút (nodes) vào đồ thị
    for node in data['nodes']:
        node_id = node['id']
        label_text = node['name']
        group = node['label']
        title = f"ID: {node_id}\nLabel: {group}"
        for key, value in node.items():
            if key not in ['id', 'name', 'label']:
                title += f"\n{key}: {value}"
        net.add_node(node_id, label=label_text, group=group, title=title)

    # 4. Thêm các cạnh (edges) vào đồ thị
    for edge in data['edges']:
        source = edge['source']
        target = edge['target']
        # Lấy tất cả thông tin của cạnh để làm tooltip
        edge_title = f"Type: {edge.get('type', '')}"
        for key, value in edge.items():
            if key not in ['source', 'target', 'type']:
                edge_title += f"\n{key}: {value}"

        net.add_edge(source, target, label=edge.get('type', ''), title=edge_title)

    # 5. Lưu đồ thị ra file HTML
    try:
        net.save_graph(output_filename)
        print(f"✅ Đồ thị đã được lưu thành công vào file '{output_filename}'")
        print("Mở file này bằng trình duyệt để xem đồ thị tương tác.")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi lưu file: {e}")

# Chạy chương trình
if __name__ == "__main__":
    visualize_knowledge_graph(json_data)