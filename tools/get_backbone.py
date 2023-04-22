import torch

def get_backbone(checkpoint_path):
    print("Get backbone")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    backbone = checkpoint['state_dict_backbone']
    torch.save(backbone, "face_reg_torch/models/backbone_weight_1.pt")
    print("DONE") 

if __name__ == '__main__':
    get_backbone("face_reg_torch/models/checkpoint_step_265000.pt")
