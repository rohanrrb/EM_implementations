import pandas as pd
import random

#asked LLM to generate some synthetic data (already blocked) for testing
anchors = [
    ("A1", "Apple iPhone 12 128GB Black"),
    ("A2", "Samsung Galaxy S21"),
    ("A3", "Sony WH-1000XM4 Headphones"),
    ("A4", "Dell XPS 13 Laptop"),
    ("A5", "Nintendo Switch OLED")
]

candidates_pool = {
    "A1": [
        ("iPhone 12 - 128 GB, black, unlocked", 1),
        ("Apple Watch Series 6", 0),
        ("iPhone 13 Pro Max 256GB", 0),
        ("Apple iPhone 12 Black 128GB", 1),
        ("MacBook Air M1", 0)
    ],
    "A2": [
        ("Galaxy S21 smartphone", 1),
        ("Samsung Galaxy S20 FE", 0),
        ("Samsung Galaxy Buds+", 0),
        ("Galaxy S21 Ultra", 1),
        ("Samsung Smart TV", 0)
    ],
    "A3": [
        ("Sony WH1000XM4 Wireless Headphones", 1),
        ("Bose QuietComfort 45", 0),
        ("Sony WF-1000XM4 Earbuds", 0),
        ("Sony WH-1000XM3", 1),
        ("JBL Live 660NC", 0)
    ],
    "A4": [
        ("Dell XPS 13 9310", 1),
        ("MacBook Pro 14-inch", 0),
        ("Dell Inspiron 15", 0),
        ("Dell XPS 13 Laptop 11th Gen", 1),
        ("HP Spectre x360", 0)
    ],
    "A5": [
        ("Nintendo Switch OLED Model", 1),
        ("PlayStation 5", 0),
        ("Nintendo Switch Lite", 0),
        ("Nintendo Switch with Neon Joy-Con", 1),
        ("Xbox Series X", 0)
    ]
}

rows = []
for anchor_id, anchor_text in anchors:
    for cand_text, label in candidates_pool[anchor_id]:
        rows.append({
            "id_left": anchor_id,
            "record_left": anchor_text,
            "record_right": cand_text,
            "label": label
        })

df = pd.DataFrame(rows)
dataset_path = "synthetic/sample_llm4em_dataset.csv"
df.to_csv(dataset_path, index=False)
dataset_path

