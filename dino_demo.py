import torch
from torch import nn
import timm

class dino_classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight="dinov2_vitb14_reg4_pretrain.pth"
        self.barlow_weight = torch.load(self.weight)
        
        self.fc = nn.Linear(self.input_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    

if __name__=="__main__":
    model=timm.create_model('vit_base_patch16_224', pretrained=False)

    weights="dinov2_vitb14_reg4_pretrain.pth"
    # dino_model=torch.load(weights)
    # model.load_state_dict(dino_model)
    # model.eval()
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    BACKBONE_SIZE="small"
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    input=torch.rand(1, 3, 224, 224)	
    import pdb;pdb.set_trace()
    backbone_model.eval()
    backbone_model.cuda()
    import pdb;pdb.set_trace()