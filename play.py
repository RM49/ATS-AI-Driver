import time
import os
import numpy as np
from pynput.keyboard import Key, Controller, Listener
from PIL import Image
from collections import deque
from keras.models import load_model
import win32gui
import ctypes
import win32ui

from keras import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense

# --- Configuration & Setup ---
MODEL_PATH = "./ats_driver_model_v2.h5"
FRAME_STACK_SIZE = 4
GAME_WINDOW = {"top": 40, "left": 0, "width": 1280, "height": 720}

# Initialize keyboard controller
keyboard = Controller()

# --- Load the Trained Model ---
model = Sequential([
    Input(shape=(66, 200, 4)),  # Input Layer for stacked frames

    # Convolutional "Eyes"
    Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
    Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
    Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    
    Flatten(),
    
    # Dense "Brain"
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    
    # --- THIS IS THE CORRECTION ---
    # Output layer now has 2 neurons
    Dense(2) 
    # ------------------------------
])

print("Loading trained model...")
model.load_weights(MODEL_PATH)
print("Model loaded successfully.")

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

play = False

def main_loop():
    # --- State Management ---
    # Use a deque for efficient frame stacking
    frame_stack = deque(maxlen=FRAME_STACK_SIZE)
    
    # Initialize a "kill switch"
    running = True
    def on_press(key):
        nonlocal running
        global play
        if key == Key.esc: # Use a key like 'esc' to stop the bot
            running = False
            return False # Stop the listener
        if key == Key.ctrl_l:
            play = not play
            keyboard.release('w')
            keyboard.release('a')
            keyboard.release('d')
    
    listener = Listener(on_press=on_press)
    listener.start()

    print("Starting AI in 3 seconds... Click on the game window!")
    time.sleep(3)

    while running:
        loop_start_time = time.time()
        # 1. Capture and Preprocess
        
        # --- Capture Screen ---
        img = capture_window_alt(hwnd)
        
        # --- Preprocess Image (using your function) ---
        processed_frame = preprocess_image(img) # Assuming you have this function
        
        # 2. Update Frame Stack
        frame_stack.append(processed_frame)
        
        current_angle = 0.0
        current_throttle = 0.0
        
        # 3. Predict (only if we have a full stack of frames)
        if len(frame_stack) == FRAME_STACK_SIZE:
            # Prepare model input: stack frames and add batch dimension
            model_input = np.concatenate(list(frame_stack), axis=-1)
            model_input = np.expand_dims(model_input, axis=0) # Shape: (1, 66, 200, 4)
            # Get model's prediction
            prediction = model.predict(model_input, verbose=0)[0]
            steering_angle = prediction[0]
            throttle = prediction[1]
            # 4. Act in the Game
            # Throttle control
            if play:
                print(f"PLAY ON | Steering: {steering_angle:.2f}, Throttle: {throttle:.2f}", end='\r')
                if throttle > 0.8 and current_throttle <= throttle:
                    keyboard.press('w')
                    current_throttle = 1.0
                elif throttle < -0.8 and current_throttle >= throttle:
                    keyboard.press('s')
                    current_throttle = -1.0
                else:
                    current_throttle = 0.0
                    keyboard.release('s')
                    keyboard.release('w')
                # Steering control
                if (steering_angle < -0.15 and current_angle >= steering_angle) or (steering_angle > 0.0 and (current_angle > steering_angle + 0.2)): # Turning left
                # if steering_angle > 0.0 and current_angle < steering_angle - 0.05:
                    keyboard.press('a')
                    keyboard.release('d')
                    current_angle += 0.1
                elif (steering_angle > 0.15 and current_angle <= steering_angle) or (steering_angle < 0.0 and (current_angle < steering_angle - 0.2)): # Turning right
                # elif current_angle > steering_angle + 0.05:
                    keyboard.press('d')
                    keyboard.release('a')
                    current_angle -= 0.1
                else: # Driving straight
                    current_angle *= 0.95
                    keyboard.release('a')
                    keyboard.release('d')
            else:
                print(f"PLAY OFF | Steering: {steering_angle:.2f}, Throttle: {throttle:.2f}", end='\r')
                current_angle = 0.0
                current_throttle = 0.0
        # Control loop speed
        # time.sleep(max(1./90 - (time.time() - loop_start_time), 0)) # Aim for ~15 FPS
        # changing to a higher fps to see if that avoids oversteering...
        # perhaps a 60fps and then keep decreasing steering angle condition until it smooth?

    print("Stopping AI.")
    listener.stop()
    # Release all keys as a safety measure
    keyboard.release('w')
    keyboard.release('a')
    keyboard.release('d')
    keyboard.release('s')


if __name__ == '__main__':
    main_loop()