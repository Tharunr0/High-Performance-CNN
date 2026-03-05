import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "data/raw/lung_colon_image_set" 
MODEL_SAVE_PATH = "models/best_model.pth"

BATCH_SIZE = 32         
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_WORKERS = 4          
PIN_MEMORY = True        
NUM_CLASSES = 5
IMAGE_SIZE = 224