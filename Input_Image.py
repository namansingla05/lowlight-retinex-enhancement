import cv2
import numpy as np

def average_brightness(image):
    """Calculate the average brightness of an image"""
    return np.mean(image)

def discrete_entropy(image):
    """Calculate the discrete entropy of an image"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    return entropy

def calculate_niqe(img, patch_size=64):
    """
    Calculate a simplified NIQE-like score for the given image
    """
    # Convert to float
    img = img.astype(np.float32) / 255.0
    
    # Calculate local statistics
    mean_local = cv2.GaussianBlur(img, (7, 7), 1.166)
    var_local = cv2.GaussianBlur(img**2, (7, 7), 1.166) - mean_local**2
    
    # Avoid division by zero
    var_local[var_local < 1e-10] = 1e-10
    
    # Normalize the image
    img_norm = (img - mean_local) / np.sqrt(var_local)
    
    # Calculate first and second order statistics
    mu = np.mean(img_norm)
    sigma = np.std(img_norm)
    
    # Calculate skewness and kurtosis
    skewness = np.mean((img_norm - mu)**3) / (sigma**3)
    kurtosis = np.mean((img_norm - mu)**4) / (sigma**4) - 3
    
    # Simple statistical distance
    ref_mu = 0.0      # Reference mean
    ref_sigma = 1.0   # Reference std
    ref_skewness = 0.0  # Reference skewness
    ref_kurtosis = 3.0  # Reference kurtosis
    
    # Calculate distance
    diff = np.array([mu - ref_mu, sigma - ref_sigma, skewness - ref_skewness, kurtosis - ref_kurtosis])
    dist = np.sqrt(diff.dot(diff.T))
    
    return dist

def calculate_image_metrics(image_path):
    """Calculate and print AB, DE, and NIQE metrics for an input image"""
    # Load image
    img = cv2.imread("img6.jpg")
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    ab = average_brightness(img_gray)
    de = discrete_entropy(img_gray)
    niqe = calculate_niqe(img_gray)
    
    # Print results
    print(f"Image: {image_path}")
    print(f"AB (Average Brightness): {ab:.4f}")
    print(f"DE (Discrete Entropy): {de:.4f}")
    print(f"NIQE: {niqe:.4f}")
    
    return ab, de, niqe

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the image file: ")
    
    calculate_image_metrics(image_path)