import zipfile
import os

dir = f"{os.getcwd()}/datasets"

# with zipfile.ZipFile(f"{dir}/celeba.zip", "r") as zip_ref:
#     zip_ref.extractall(dir)

with zipfile.ZipFile(f"{dir}/celeba/img_align_celeba.zip", "r") as zip_ref:
     zip_ref.extractall(f"{dir}/celeba/")