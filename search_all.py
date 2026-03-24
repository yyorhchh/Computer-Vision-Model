import os

files = [
    'c:/cv/Week5 Part1 Complete_Blood_Count.ipynb',
    'c:/cv/Week5 Part1 Complete_Blood_Count done.ipynb',
    'c:/Users/Asus/Downloads/Week5 Part1 Complete_Blood_Count1.ipynb',
    'c:/Users/Asus/Downloads/Week5 Part1 Complete_Blood_Count.ipynb',
    'c:/Users/Asus/Downloads/Week4 Part2 Deep_Computer_Vision_PyTorch.ipynb',
    'c:/Computer vision/index_performance.html'
]

search_strings = ['CustomCNN', 'AdvancedModel', 'nn.Conv2d', 'best_custom_cnn.pth', 'best_resnet_advanced.pth', 'best_resnet50.pth']

for f in files:
    if os.path.exists(f):
        try:
            found = set()
            with open(f, encoding='utf-8') as file:
                for line in file:
                    for s in search_strings:
                        if s.lower() in line.lower():
                            found.add(s)
            
            if found:
                print(f"FOUND in {f}: {list(found)}")
            else:
                print(f"NOT FOUND in {f}")
        except Exception as e:
            print(f"ERROR reading {f}: {e}")
    else:
        print(f"FILE NOT FOUND: {f}")
