import numpy as np

def calculate_distance(tensor1, tensor2):
    # Assuming tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"

    # Calculate Euclidean distance for each pair of joints
    distances = np.linalg.norm(tensor1 - tensor2, axis=-1)

    return distances



if __name__ == "__main__":
    # Example usage:
    gt_path = "text2motion/data/GRAB/joints/s1/airplane_fly_1.npy"
    prediction_path = "/dtu/blackhole/13/181395/MotionDiffuse/text2motion/checkpoints/grab/md_alexExp2fixedseed_seed42"
    tensor1 = np.load(gt_path)  
    tensor2 = np.array(prediction_path)  

    distances = calculate_distance(tensor1, tensor2)

    print("Distances for each joint:", distances)
