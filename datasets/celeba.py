import gdown
import os

url = "https://drive.google.com/uc?id=1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ"
output = f"{os.getcwd()}/datasets/celeba.zip" 


gdown.download(url, output, quiet=False)