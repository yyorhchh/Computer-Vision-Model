import json
import os

keywords = ["CustomCNN", "AdvancedModel", "best_custom_cnn", "best_resnet_advanced", "best_resnet50"]
search_dirs = [r"c:\Computer vision", r"c:\cv", r"c:\Users\Asus\Downloads"]

for d in search_dirs:
    if not os.path.exists(d):
        continue
    print(f"Searching in {d}...")
    for f in os.listdir(d):
        if f.endswith(".ipynb"):
            path = os.path.join(d, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    for i, cell in enumerate(data.get("cells", [])):
                        source = "".join(cell.get("source", []))
                        for kw in keywords:
                            if kw.lower() in source.lower():
                                print(f"Found '{kw}' in {path} at cell {i}")
                                # Print a snippet of the code
                                lines = source.split("\n")
                                for line in lines:
                                    if kw.lower() in line.lower():
                                        print(f"  Line: {line.strip()}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
