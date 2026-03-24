import torch
import collections

def inspect_model(path):
    print(f"Inspecting {path}...")
    try:
        # Load state dict
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, collections.OrderedDict):
            print("Detected state_dict (OrderedDict)")
            for key in list(checkpoint.keys())[:10]:
                print(f"  {key}: {checkpoint[key].shape}")
        else:
            print(f"Detected object of type: {type(checkpoint)}")
            # If it's a full model, we might see the class name here
            print(f"Class name: {checkpoint.__class__.__name__}")
            if hasattr(checkpoint, 'layers'):
                print("Detected layers attribute")
    except Exception as e:
        print(f"Error loading {path}: {e}")

inspect_model(r"C:\Computer vision\model\best_custom_cnn.pth")
print("-" * 20)
inspect_model(r"C:\Computer vision\model\best_resnet50.pth")
