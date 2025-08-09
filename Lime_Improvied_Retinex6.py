import cv2
import numpy as np
from scipy.ndimage import sobel, laplace, gaussian_filter
from scipy import special

def compute_gradients(image, method='sobel'):
    if method == 'sobel':
        Gx = sobel(image, axis=1)
        Gy = sobel(image, axis=0)
        G = np.sqrt(Gx**2 + Gy**2)
    elif method == 'laplacian':
        G = laplace(image)
    else:
        raise ValueError("Invalid gradient method. Choose 'sobel' or 'laplacian'.")
    D = np.abs(G)
    return G, D

def iterative_least_squares(V, alpha, beta, iterations=20, method='sobel'):
    V = V.astype(np.float32) / 255.0
    T = np.copy(V)
    R = np.ones_like(V)

    for k in range(iterations):
        G, D = compute_gradients(T, method)
        T = (V * R) / (R + alpha * G + 1e-6)
        R = (V * T) / (T + beta * D + 1e-6)

    return T, R

# Image quality metrics functions
def average_brightness(image):
    """Calculate the average brightness of an image"""
    if len(image.shape) == 3:
        # Convert color image to grayscale for brightness calculation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return np.mean(gray)

def discrete_entropy(image):
    """Calculate the discrete entropy of an image"""
    if len(image.shape) == 3:
        # Convert color image to grayscale for entropy calculation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    return entropy

def calculate_niqe(img, patch_size=64):
    """
    Calculate a simplified NIQE-like score for the given image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
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
    # For a true NIQE implementation, we'd compare to a trained natural model
    # Here we're using fixed reference values for simplicity
    ref_mu = 0.0  # Reference mean (ideally 0 for normalized natural images)
    ref_sigma = 1.0  # Reference std (ideally 1 for normalized natural images)
    ref_skewness = 0.0  # Reference skewness (ideally 0 for natural images)
    ref_kurtosis = 3.0  # Reference kurtosis (ideally 3 for natural images)
    
    # Calculate Mahalanobis-like distance
    diff = np.array([mu - ref_mu, sigma - ref_sigma, skewness - ref_skewness, kurtosis - ref_kurtosis])
    
    # Compute distance (simplified version of equation in your image)
    dist = np.sqrt(diff.dot(diff.T))
    
    return dist

def enhance_low_light_image(image_path, iterations=20, alpha=0.001, beta=0.0001, method='sobel', 
                           preserve_colors=True, denoise=True):
    """Enhanced version that preserves colors better and reduces noise"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Optional pre-denoising for very dark images
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # Original RGB for later color preservation
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get input image metrics before enhancement
    input_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ab_input = average_brightness(input_gray)
    de_input = discrete_entropy(input_gray)
    niqe_input = calculate_niqe(input_gray)
    
    # Process in HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    
    # Analyze image and set parameters adaptively
    mean_V = np.mean(V) / 255.0
    if alpha is None:
        alpha = 0.001 + 0.002 * (0.5 - mean_V)  # Darker images need higher alpha
        alpha = max(0.0001, min(0.003, alpha))  # Limit range to prevent artifacts
    
    if beta is None:
        beta = 0.0001 + 0.0003 * (0.5 - mean_V)
        beta = max(0.00005, min(0.0005, beta))  # Limit range
    
    print(f"Using alpha: {alpha:.6f}, beta: {beta:.6f}")
    
    # Apply the core enhancement algorithm
    T, R = iterative_least_squares(V, alpha, beta, iterations, method)
    
    # Adaptive gamma correction
    mean_intensity = np.mean(T)
    gamma = 1.0 if mean_intensity >= 0.5 else 1.0 + (0.5 - mean_intensity) * 2.0
    gamma = min(2.2, gamma)  # Cap gamma to prevent overexposure
    
    T_corrected = np.power(T, 1/gamma)
    
    # Apply brightness enhancement to V channel
    #enhanced_V = np.clip(T_corrected * 255, 0, 255).astype(np.uint8)
    # Apply CLAHE after gamma correction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    T_uint8 = np.clip(T_corrected * 255, 0, 255).astype(np.uint8)
    T_clahe = clahe.apply(T_uint8)

    enhanced_V = T_clahe  # Use CLAHE output instead

    
    # Better color preservation method - only update V channel
    if preserve_colors:
        # Create enhanced image directly in RGB
        enhanced_hsv = hsv.copy()
        enhanced_hsv[:, :, 2] = enhanced_V
        enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        # Reduce color saturation for overexposed areas
        enhanced_hsv_normalized = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
        
        # Identify bright areas
        bright_mask = (enhanced_V > 200).astype(np.float32)
        bright_mask = cv2.GaussianBlur(bright_mask, (0, 0), sigmaX=3)
        
        # Reduce saturation in bright areas to prevent color shifts
        enhanced_hsv_normalized[:, :, 1] = enhanced_hsv_normalized[:, :, 1] * (1 - 0.5 * bright_mask)
        enhanced_rgb = cv2.cvtColor(enhanced_hsv_normalized, cv2.COLOR_HSV2RGB)
    else:
        # Simple HSV to RGB conversion without color correction
        hsv[:, :, 2] = enhanced_V
        enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Optional post-processing to reduce noise in dark areas
    if denoise:
        # Create mask for dark areas
        dark_mask = (V < 50).astype(np.uint8)
        dark_mask = cv2.dilate(dark_mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # Apply stronger denoising to dark areas
        denoised = cv2.fastNlMeansDenoisingColored(enhanced_rgb, None, 5, 5, 7, 21)
        
        # Blend denoised and enhanced image based on dark mask
        dark_mask_float = dark_mask.astype(float) / 255
        dark_mask_float = np.stack([dark_mask_float] * 3, axis=2)
        enhanced_rgb = enhanced_rgb * (1 - dark_mask_float) + denoised * dark_mask_float
    
    # Get enhanced image metrics
    enhanced_result = enhanced_rgb.astype(np.uint8)
    enhanced_gray = cv2.cvtColor(enhanced_result, cv2.COLOR_RGB2GRAY)
    ab_output = average_brightness(enhanced_gray)
    de_output = discrete_entropy(enhanced_gray)
    niqe_output = calculate_niqe(enhanced_gray)
    
    # Print metrics
    print(f"Enhanced Image - AB (Average Brightness): {ab_output:.4f}, DE (Discrete Entropy): {de_output:.4f}, NIQE: {niqe_output:.4f}")
    print(f"Computed Gamma: {gamma:.2f}")
    
    return original_rgb, enhanced_rgb.astype(np.uint8)

def resize_image(image, width=600):
    h, w = image.shape[:2]
    scaling_factor = width / float(w)
    height = int(h * scaling_factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def show_images_side_by_side(original, enhanced):
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    original_resized = resize_image(original_bgr, width=600)
    enhanced_resized = resize_image(enhanced_bgr, width=600)

    combined = np.hstack((original_resized, enhanced_resized))

    cv2.imshow('Original (Left) vs Enhanced (Right)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_comparison(original, enhanced, output_path="enhanced_result.jpg"):
    """Save the comparison to file"""
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    original_resized = resize_image(original_bgr, width=600)
    enhanced_resized = resize_image(enhanced_bgr, width=600)

    combined = np.hstack((original_resized, enhanced_resized))
    cv2.imwrite(output_path, combined)
    print(f"Result saved to {output_path}")

# Example usage 
if __name__ == "__main__":
    image_path = 'img8.png'  # Replace with your image file
    original, enhanced = enhance_low_light_image(
        image_path, 
        iterations=20,
        preserve_colors=True,
        denoise=True
    )
    show_images_side_by_side(original, enhanced)
    save_comparison(original, enhanced)
