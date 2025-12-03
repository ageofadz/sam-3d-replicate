import cv2
import numpy as np
import argparse

rect = None
drawing = False
start_point = None

def draw_rectangle(event, x, y, flags, param):
    global rect, drawing, start_point, img_display, img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = img.copy()
        cv2.rectangle(img_display, start_point, (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (min(start_point[0], x), min(start_point[1], y), 
                abs(x - start_point[0]), abs(y - start_point[1]))
        cv2.rectangle(img_display, start_point, (x, y), (0, 255, 0), 2)

def generate_mask(image_path: str, output_path: str = "mask.png"):
    global rect, img_display, img
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_display = img.copy()
    
    cv2.namedWindow('Select Object')
    cv2.setMouseCallback('Select Object', draw_rectangle)
    
    print("Instructions:")
    print("- Draw a rectangle around the object")
    print("- Press ENTER to segment")
    print("- Press 'r' to redraw rectangle")
    print("- Press 'q' to quit")
    
    while True:
        cv2.imshow('Select Object', img_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 and rect is not None:
            x, y, w, h = rect
            img_h, img_w = img.shape[:2]
            
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))
            
            if w < 2 or h < 2:
                print("Rectangle too small. Please draw a larger rectangle.")
                continue
            
            rect_clamped = (x, y, w, h)
            
            mask = np.zeros(img.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            try:
                cv2.grabCut(img, mask, rect_clamped, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            except cv2.error as e:
                print(f"Error during segmentation: {e}")
                print("Please try drawing a larger rectangle around the object.")
                continue
            
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = img * mask2[:, :, np.newaxis]
            
            mask_save = mask2 * 255
            cv2.imwrite(output_path, mask_save)
            print(f"Saved mask to {output_path}")
            
            cv2.imshow('Result', result)
            cv2.imshow('Mask', mask_save)
            cv2.waitKey(2000)
            break
        elif key == ord('r'):
            rect = None
            img_display = img.copy()
        elif key == ord('q'):
            print("Cancelled")
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a mask for an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="mask.png", help="Path to output mask (default: mask.png)")
    
    args = parser.parse_args()
    generate_mask(args.image, args.output)

