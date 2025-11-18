import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
import os

# dir_path = os.path.dirname(os.path.realpath(__file__))
# image_dir = f"{dir_path}/celeba/img_align_celeba/"  # path to images
# output_file = "celeba_dataset.pt"  # single file to store everything
# all_images = []
# image_files = sorted(os.listdir(image_dir))  # important: consistent order
# # Transformation
# image_size = 64
# transform = transforms.Compose([
#     transforms.CenterCrop(178),
#     transforms.Resize((image_size, image_size)),
#     transforms.ToTensor(),  # Converts to [0, 1] range
# ])
# batch_size = 2000
# for batch_idx in tqdm(range(0, len(image_files), batch_size)):
#     batch_tensors = []
#     for img_file in image_files[batch_idx:batch_idx + batch_size]:
#         img = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
#         batch_tensors.append(transform(img))

#     batch_tensor = torch.stack(batch_tensors)  # [B, 3, H, W]
#     file = f"celeba_batch_{batch_idx // batch_size:03d}.pt"
#     torch.save(batch_tensor, os.path.join(f"{dir_path}/celeba/64x64/", file))


#     del batch_tensor

# file = f"celeba_batch_{1:03d}.pt"
# data = torch.load(os.path.join(f"{dir_path}/celeba/", file), weights_only=False)
# print(data.shape)
# save_image(data[:100], "test.png", nrow=10)

# file = f"celeba_batch_{1:03d}.pt"
# data = torch.load(os.path.join(f"{dir_path}/celeba/64x64/", file), weights_only=False)
# print(data.shape)
# save_image(data[:100], "test_64_64.png", nrow=10)

folder = "/home/hadziarm/vqvae-tpm/datasets/celeba"
input_folder = f"{folder}/64x64"
output_folder = f"{folder}/32x32"

transform = transforms.Compose([
    transforms.Resize((32, 32)),
])



for filename in os.listdir(input_folder):
    if filename.endswith(".pt") or filename.endswith(".pth"):
        file_path = os.path.join(input_folder, filename)

        # Load tensor
        tensor = torch.load(file_path, weights_only=False)
        print(tensor.shape)
        # Apply transform to each image in batch
        downscaled = torch.stack([transform(img) for img in tensor])
        
        # # Save reshaped tensor in output folder
        out_path = os.path.join(output_folder, filename)
        torch.save(downscaled, out_path)

        print(f"Processed {filename}: original {tuple(tensor.shape)} → new {tuple(downscaled.shape)}")
