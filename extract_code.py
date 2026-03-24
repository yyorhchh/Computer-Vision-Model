import os
import json

def search_notebooks(directories, search_strings):
    found_info = []
    print(f"Searching in: {directories}")
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
        print(f"Scanning directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.ipynb'):
                    path = os.path.join(root, file)
                    print(f"Checking file: {path}")
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            nb = json.load(f)
                            cells = nb.get('cells', [])
                            # print(f"  Found {len(cells)} cells")
                            for i, cell in enumerate(cells):
                                source = cell.get('source', [])
                                if isinstance(source, list):
                                    content = "".join(source)
                                else:
                                    content = str(source)
                                
                                for s in search_strings:
                                    if s.lower() in content.lower():
                                        print(f"    Match found for {s} in cell {i}")
                                        found_info.append(f"FILE: {path}\n--- Cell {i} ---\nSTRING: {s}\nCONTENT:\n{content[:1000]}...\n")
                                        break
                    except Exception as e:
                        print(f"Error reading {path}: {e}")
    
    with open('c:/Computer vision/found_classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(found_info))
    print(f"Found {len(found_info)} matches. Results written to found_classes.txt")

if __name__ == "__main__":
    search_notebooks(['c:/cv', 'c:/Computer vision', 'c:/Users/Asus/Downloads'], ['CustomCNN', 'AdvancedModel', 'nn.Conv2d', 'best_custom_cnn.pth', 'best_resnet_advanced.pth'])
