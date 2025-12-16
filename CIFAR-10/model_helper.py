import timm
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms  
 

ADAEA_MODELS_DICT = {
    "resnetv2_101x1_bitm": "target/resnetv2_101x1_bitm.pth.tar",
    "vit_base_patch16_224": "target/vit_base.pth.tar",
    "swin_base_patch4_window7_224": "target/swin_base.pth.tar",
    "deit_base_patch16_224": "target/deit_base.pth.tar", 

    "deit_tiny_patch16_224": "deit_tiny.pth.tar",
    "vit_tiny_patch16_224": "vit_tiny.pth.tar"
}

OUR_MODELS_DICT = {
    "convmixer_768_32": "vit", 
    "regnety_160": "vit", 
    "resnet152": "resnet152", 
    "efficientnet_b0": "vit", 
    "resnet18": "resnet18", 
    "inception_v3": "inception_v3", 
    "gcvit_tiny": "vit"
} 

save_root_path = r"Models"


class TransformWrapper(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        # x is a tensor of shape [B, C, H, W] from DataLoader
        # If needed, convert PIL Image to tensor first
        # For CIFAR-10 DataLoader, it's already tensor
        return self.transform(x)
        
def wrap_model_with_transform(model_name, model):
    if any(x in model_name for x in ['inc', 'bit']):
        transform = nn.Sequential(
            TransformWrapper(transforms.Resize((299, 299))),
            TransformWrapper(transforms.Normalize((0.5,), (0.5,)))
        )
    elif any(x in model_name for x in ['vit', 'deit', 'swin']):
        transform = nn.Sequential(
            TransformWrapper(transforms.Resize((224, 224))),
            TransformWrapper(transforms.Normalize((0.5,), (0.5,)))
        )
    else:
        transform = nn.Sequential(
            TransformWrapper(transforms.Resize((224, 224))),
            TransformWrapper(transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010)))
        )
    return nn.Sequential(transform, model)



def get_models(dataset, model_name, key):
    if dataset == 'cifar10':
        # For Inception, resize to 299x299; otherwise 32x32 is fine
        if any(x in key for x in ['inc', 'bit']):
            transform_resize = transforms.Resize((299, 299))
            norm = transforms.Normalize((0.5,), (0.5,)) 
        elif any(x in key for x in ['vit']):
            transform_resize = transforms.Resize((224, 224))
            norm = transforms.Normalize((0.5,), (0.5,)) 
        else:
            transform_resize = transforms.Resize((224, 224)) # do nothing
            # Standard CIFAR-10 normalization
            norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010)) 
    
        # Create model
        model = timm.create_model(model_name, pretrained=True, num_classes=10)
        model.eval()
        
        # Wrap resize + normalization + model in Sequential
        return torch.nn.Sequential(
            transform_resize,   # resize if Inception
            norm,               # normalization
            model
        )


def load_model_hub(model_name): 
    if model_name in ADAEA_MODELS_DICT: 
        checkpoint_path = os.path.join(save_root_path, ADAEA_MODELS_DICT[model_name])  
        print(f"\n🔹 Loading {model_name} from {checkpoint_path}") 
        # 1. Create model WITHOUT checkpoint 
        model = timm.create_model(model_name, pretrained=False, num_classes=10)
        # 2. Load checkpoint dictionary
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Checkpoint may contain "state_dict"
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    
        # 3. Remove 'module.' prefix
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
        # 4. Load weights
        msg = model.load_state_dict(clean_sd, strict=False)
        print("Load result:", msg)
    
        return wrap_model_with_transform(model_name, model.eval())  
        
    elif model_name in OUR_MODELS_DICT:
        model = get_models('cifar10', model_name, OUR_MODELS_DICT[model_name])  
        state_dict = torch.load(f"checkpoints/{model_name}_cifar10.pth")
        model_state = model.state_dict()
        model.load_state_dict(state_dict['model_state'])

        return model.eval() 
