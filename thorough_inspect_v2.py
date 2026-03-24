import torch
import os
import collections

def thorough_inspect(path):
    print(f"--- Thorough Inspection of {os.path.basename(path)} ---")
    if not os.path.exists(path):
        print(f"Error: File {path} does not exist.")
        return
    try:
        data = torch.load(path, map_location='cpu')
        
        if isinstance(data, (dict, collections.OrderedDict)):
            print(f"Checkpoint is a dictionary with {len(data)} keys.")
            for k in list(data.keys()):
                val = data[k]
                if isinstance(val, torch.Tensor):
                    print(f"Layer: {k} | Shape: {val.shape}")
                elif isinstance(val, (int, float, str, bool, list, dict)):
                    if not isinstance(val, (list, dict)) or len(str(val)) < 200:
                        print(f"Key: {k} | Value: {val}")
                else:
                    print(f"Key: {k} | Type: {type(val)}")
        else:
            print(f"Data is not a dict. Type: {type(data)}")
            # If it's a full model object, show its structure
            print(data)
            
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

models = ['model/best_custom_cnn.pth', 'model/best_resnet50.pth']
for m in models:
    thorough_inspect(os.path.join('c:\\Computer vision', m))
