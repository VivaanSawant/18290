import numpy as np
import matplotlib.pyplot as plt
import cv2

def conv2d_same(img, kernel, pad_mode="edge"):
    """
    2D convolution (not correlation), output same size as img.
    pad_mode: "edge" replicates boundary pixels.
    """
    img = img.astype(np.float64)
    k = kernel.astype(np.float64)

    # Flip kernel for convolution
    k = np.flipud(np.fliplr(k))

    H, W = img.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode)
    out = np.zeros((H, W), dtype=np.float64)

    for m in range(H):
        for n in range(W):
            region = padded[m:m+kh, n:n+kw]
            out[m, n] = np.sum(region * k)

    return out


def load_grayscale(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    img = img.astype(np.float64)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2) 
    return img


def save_side_by_side(img1, img2, title1, title2, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()


def save_single(img, title, outpath, normalize=False):
    disp = img
    if normalize:
        disp = disp / (disp.max() + 1e-12)

    plt.figure(figsize=(5, 5))
    plt.imshow(disp, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
   
    IMG_PATH = "image.png"  

    img = load_grayscale(IMG_PATH)

 
    K = 9 
    box = np.ones((K, K), dtype=float) / (K * K)
    img_blur = conv2d_same(img, box, pad_mode="edge")

    save_side_by_side(
        img, img_blur,
        "Original Image",
        f"Blurred Image (Box {K}x{K})",
        "problem4b_blur.png"
    )

   
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)
    Sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)

    Gx = conv2d_same(img, Sx, pad_mode="edge")
    Gy = conv2d_same(img, Sy, pad_mode="edge")
    Gmag = np.sqrt(Gx**2 + Gy**2)

    save_single(Gx, "Sobel Output: Gx", "problem4d_Gx.png", normalize=False)
    save_single(Gy, "Sobel Output: Gy", "problem4d_Gy.png", normalize=False)
    save_single(Gmag, "Gradient Magnitude", "problem4d_Gmag.png", normalize=True)

    print("\n--- Part (d) Answers ---")
    print("(a) Gx strongest edges: VERTICAL edges, because Sx computes left-to-right intensity differences (∂/∂n).")
    print("(b) Gy strongest edges: HORIZONTAL edges, because Sy computes top-to-bottom intensity differences (∂/∂m).")
    print("(c) Gradient magnitude highlights edges regardless of direction because it combines both components: sqrt(Gx^2 + Gy^2).")
    print("(d) Box blur suppresses high-frequency detail (averaging smooths rapid pixel changes). Sobel emphasizes high-frequency detail (derivative-like operation boosts rapid changes).")

  
    dm, dn = 20, 30    
    alpha = 1.0     

    kernel = np.zeros((dm + 1, dn + 1), dtype=float)
    kernel[dm, dn] = 1.0
    kernel[0, 0] = alpha

    y_translate = conv2d_same(img, kernel, pad_mode="edge")

    save_side_by_side(
        img, y_translate,
        "Original Image",
        f"Two Copies (shift = ({dm},{dn}), alpha = {alpha})",
        "problem4e_translate.png"
    )

    print("\n--- Part (e) Translation Kernel h[k, l] (matrix you can submit) ---")
    print(kernel)
