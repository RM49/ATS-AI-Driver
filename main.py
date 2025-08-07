import win32gui
import win32ui
import win32con
import ctypes  # Import ctypes
from PIL import Image, ImageShow
import time
import numpy as np
import os
import csv
from pynput import keyboard
# --- The Fix ---
# Add this line to make the script DPI-aware
ctypes.windll.user32.SetProcessDPIAware()
# ---------------

# window screenshot dimensions
screenshot_width = 1310
screenshot_height = 795
count = 0

def capture_window_alt(hwnd):
    """
    Captures a window's contents using the PrintWindow API.
    This version is DPI-aware and captures the entire window.
    """
    try:
        # Get the complete window rectangle, including the frame, in physical pixels.
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        # Get the window's device context
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create a bitmap object
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # Call PrintWindow. The 3rd parameter (2) is PW_RENDERFULLCONTENT.
        # This captures the entire window content, including frames and UWP elements.
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)

        # Get the bitmap's bits
        bmp_info = save_bitmap.GetInfo()
        bmp_str = save_bitmap.GetBitmapBits(True)

        # Create a PIL Image from the raw bitmap data
        image = Image.frombuffer(
            'RGB',
            (bmp_info['bmWidth'], bmp_info['bmHeight']),
            bmp_str, 'raw', 'BGRX', 0, 1)

        # Clean up resources
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        return image

    except Exception as e:
        # Clean up resources in case of an error
        if 'save_bitmap' in locals() and save_bitmap.GetHandle():
            win32gui.DeleteObject(save_bitmap.GetHandle())
        if 'save_dc' in locals() and save_dc.GetSafeHdc():
            save_dc.DeleteDC()
        if 'mfc_dc' in locals() and mfc_dc.GetSafeHdc():
            mfc_dc.DeleteDC()
        if 'hwnd_dc' in locals():
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            
        print(f"Error in capture_window_alt: {e}")
        return None


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Takes a PIL Image, preprocesses it, and returns a NumPy array.
    
    1. Crops to a region of interest.
    2. Resizes the image.
    3. Converts to grayscale.
    4. Normalizes pixel values.
    """
    # --- Step 1: Crop to Region of Interest (ROI) ---
    # These coordinates are (left, top, right, bottom).
    # You MUST tune these for your specific game resolution and view.
    # The goal is to capture the road ahead, cutting out the sky and dashboard.
    # For a 1280x720 image, a good starting point might be:
    
    left = 379
    top = 222  # Cut off the sky
    right = 1293
    bottom = 530 # Cut off the dashboard
    
    cropped_img = img.crop((left, top, right, bottom))

    # --- Step 2: Resize the Image ---
    # NVIDIA's self-driving car paper used 200x66
    resized_img = cropped_img.resize((200, 66), Image.Resampling.LANCZOS)

    # --- Step 3: Convert to Grayscale ---
    grayscale_img = resized_img.convert('L')
    
    # --- Step 4: Normalize Pixel Values ---
    # Convert image to a NumPy array
    image_array = np.array(grayscale_img, dtype=np.float32)
    
    # Scale pixel values from [0, 255] to [0, 1]
    normalized_array = image_array / 255.0
    
    # Add a channel dimension for the CNN (e.g., (66, 200) -> (66, 200, 1))
    # This is required for most deep learning frameworks like TensorFlow/Keras or PyTorch
    return normalized_array[..., np.newaxis]

# --- Main Execution ---
window_title = "American Truck Simulator"
hwnd = win32gui.FindWindow(None, window_title)
# while count < 10:
#     if hwnd:
#         print(f"Found window '{window_title}' with HWND: {hwnd}")

#         # Use the new, reliable capture function
#         screenie = capture_window_alt(hwnd)

#         if screenie:
#             name = "original" + str(count) + ".png"
#             screenie.save(name)
#             preprocess_image(screenie)
#         else:
#             print("Failed to capture window.")
#     else:
#         print(f"Window '{window_title}' not found.")
#     count+=1


# --- Configuration ---
DATA_DIR = r"\\wsl.localhost\Ubuntu\home\raymi\projects\TruckDriver\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LOG_FILE = os.path.join(DATA_DIR, "log.csv")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Game window dimensions (TUNE THIS)
MONITOR = {"top": 40, "left": 0, "width": 1280, "height": 720}

# --- Keyboard State ---
keys_pressed = set()

is_collecting = False # <-- NEW: State variable to control data collection

def on_press(key):
    global is_collecting # <-- MODIFIED: Need to modify global state
    try:
        # <-- NEW: Toggle collection on/off with 'c' key
        if key.char == 'c':
            is_collecting = not is_collecting
            status = "ON" if is_collecting else "OFF"
            # Print on a new line to not interfere with the status line
            print(f"\nData collection toggled {status}")
            
        # Add driving/quit keys to the set
        if key.char in 'wasdq':
            keys_pressed.add(key.char)
    except AttributeError:
        pass # Handle special keys if needed

def on_release(key):
    try:
        keys_pressed.remove(key.char)
    except (AttributeError, KeyError):
        pass

# Start the non-blocking listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# --- Main Data Collection Loop ---
print("Starting data collection in 7 seconds...")
time.sleep(7)
print("GO!")
starttime = time.time()
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'image_filename', 'steering_angle', 'throttle'])

    steering_angle = 0.0
    throttle = 0.0
    
    run = True
    
    while True:
        # Exit condition
        if 'q' in keys_pressed:
            print("Stopping...")
            break
        
        # --- Determine Steering and Throttle ---
        if 'w' in keys_pressed:
            throttle = 1.0
        elif 's' in keys_pressed:
            throttle = -1.0
        else:
            throttle = 0.0 # maybe too much throttle 0 is being stored?
        
        if 'a' in keys_pressed:
            steering_angle -= 0.1 # Adjust step value
        elif 'd' in keys_pressed:
            steering_angle += 0.1 # Adjust step value
        else:
            # Decay towards center
            steering_angle *= 0.95 
        steering_angle = np.clip(steering_angle, -1.0, 1.0) # Keep within [-1, 1]
        
        if is_collecting:
            timestamp = time.time()
            img = capture_window_alt(hwnd)
            
            if img:
                processed_array = preprocess_image(img)
                
                # --- Save Data ---
                image_filename = f"{timestamp}.npy"
                image_path = os.path.join(IMAGES_DIR, image_filename)
                
                np.save(image_path, processed_array)
                writer.writerow([timestamp, image_filename, steering_angle, throttle])
                
                # Update status line
                print(f"REC ON | Steering: {steering_angle:.2f}, Throttle: {throttle:.2f}", end='\r')
            else:
                print("\nWindow capture failed. Pausing collection.")
                is_collecting = False # Pause collection if window is lost
        else:
            # Update status line when not recording
            print(f"REC OFF| Steering: {steering_angle:.2f}, Throttle: {throttle:.2f}", end='\r')
            # Sleep briefly to prevent the CPU from maxing out while idle
            time.sleep(0.05) 
            
        # Write to CSV log
        # writer.writerow([timestamp, image_filename, steering_angle, throttle])
        # print(f"Steering: {steering_angle:.2f}, Throttle: {throttle:.2f}")

print("time spent:" + str(time.time() - starttime))
listener.stop()