import torch

class Config:
    VOXEL_RES = 128
    WORLD_MIN = -1.0
    WORLD_MAX = 1.0
    
    NUM_VIEWS = 32
    DETECTOR_RES = 128
    DETECTOR_SIZE = 2.0
    R_ORBIT = 2.0
    
    NUM_SAMPLES_RAY = 256
    STEP_SIZE = 3.0 / 256
    
    NUM_GAUSSIANS_START = 10000
    MAX_GAUSSIANS = 30000
    DENSIFY_INTERVAL = 64
    
    TAU_GRAD = 0.0003
    TAU_OPACITY = 0.04
    TAU_SCALE = 0.1
    
    LAMBDA_L1 = 8.0e-5
    LAMBDA_VOL = 7.0e-4           
    LAMBDA_ASPECT = 1.0e-7
    
    LR_MEANS = 2.0e-2
    LR_REST = 1.0e-2
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
