# play.py (Stateful Driving)

import time
import os
import numpy as np
from pynput.keyboard import Key, Controller, Listener
from pynput import keyboard
from PIL import Image
from collections import deque
from keras.models import load_model # Use load_model for functional API models
import win32gui
import ctypes
import win32ui
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate

# --- Configuration & Setup ---
MODEL_PATH = "ats_driver_model_6x_augmented.h5"
FRAME_STACK_SIZE = 4
WINDOW_TITLE = "American Truck Simulator"

# Initialize keyboard controller
keyboard = Controller()

# --- Load the Trained Stateful Model ---
print("Loading stateful model...")
# We use load_model, which correctly reconstructs the multi-input functional model.
# No need to define the architecture here.

# --- Input Branch 1: Convolutional "Eyes" ---
image_input = Input(shape=(66, 200, 4), name='image_input') # 8 is FRAME_STACK_SIZE
x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(image_input)
x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
image_features = Flatten()(x)

# --- Input Branch 2: Previous Measurements "State" ---
measurements_input = Input(shape=(2,), name='measurements_input')

# --- Combined "Brain" ---
# Concatenate the features from the images with the measurement inputs
combined = Concatenate()([image_features, measurements_input])

# Dense layers to process the combined features
y = Dense(100, activation='relu')(combined)
y = Dense(50, activation='relu')(y)
y = Dense(10, activation='relu')(y)

# Output layer with 2 neurons for steering and throttle
output = Dense(2, name='output')(y)

# Create the final model
model = Model(inputs=[image_input, measurements_input], outputs=output)

model.load_weights(MODEL_PATH)
# --- DPI Awareness for Screen Capture ---
# This is crucial for correct window capturing on high-DPI displays
ctypes.windll.user32.SetProcessDPIAware()

def capture_window_alt(hwnd):
    """
    Captures a window's contents using the PrintWindow API.
    This version is DPI-aware and captures the entire window.
    """
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)

        bmp_info = save_bitmap.GetInfo()
        bmp_str = save_bitmap.GetBitmapBits(True)

        image = Image.frombuffer(
            'RGB',
            (bmp_info['bmWidth'], bmp_info['bmHeight']),
            bmp_str, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        return image
    except Exception as e:
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
    left = 379
    top = 222
    right = 1293
    bottom = 530
    
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((200, 66), Image.Resampling.LANCZOS)
    grayscale_img = resized_img.convert('L')
    image_array = np.array(grayscale_img, dtype=np.float32)
    normalized_array = image_array / 255.0
    return normalized_array[..., np.newaxis]

keys_pressed = set()

def main_loop():
    """
    Main loop for capturing, predicting, and controlling the vehicle.
    """
    # --- State Management ---
    frame_stack = deque(maxlen=FRAME_STACK_SIZE)
    
    # Initialize the vehicle's state. This will be updated by the model's predictions.
    current_steering = 0.0
    current_throttle = 0.0
    
    # --- Control Flags ---
    running = True
    is_ai_enabled = False
    
    def on_press(key):
        nonlocal running, is_ai_enabled
        try:
            if key == Key.esc:
                running = False
                return False # Stop the listener
            if key == Key.ctrl_l:
                is_ai_enabled = not is_ai_enabled
                status = "ON" if is_ai_enabled else "OFF"
                print(f"\n--- AI Driving {status} ---")
                # Release all keys when toggling to prevent them from getting stuck
                keyboard.release('w')
                keyboard.release('a')
                keyboard.release('d')
                keyboard.release('s')
            # Add driving/quit keys to the set
            if key.char in 'wasdq':
                keys_pressed.add(key.char)
        except (AttributeError, KeyError):
            pass
    
    def on_release(key):
        try:
            keys_pressed.remove(key.char)
        except (AttributeError, KeyError):
            pass
    
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
    if not hwnd:
        print(f"Error: Game window '{WINDOW_TITLE}' not found. Exiting.")
        running = False

    print("Starting AI in 3 seconds... Click on the game window!")
    time.sleep(3)
    print("Started OFF")

    while running:
        
        # 1. Capture and Preprocess
        img = capture_window_alt(hwnd)
        if img is None:
            print("Window capture failed. Pausing AI.")
            is_ai_enabled = False
            continue
            
        processed_frame = preprocess_image(img)
        
        # 2. Update Frame Stack
        frame_stack.append(processed_frame)
        
        # 3. Predict (only if we have a full stack of frames)
        if len(frame_stack) == FRAME_STACK_SIZE:
            # --- Prepare Model Inputs ---
            # Input 1: The stack of image frames
            image_input = np.concatenate(list(frame_stack), axis=-1)
            image_input = np.expand_dims(image_input, axis=0) # Shape: (1, 66, 200, 4)
            
            # Input 2: The previous control measurements
            measurements_input = np.array([[current_steering, current_throttle]]) # Shape: (1, 2)
            
            # --- Get Model's Prediction ---
            # The model expects a list of inputs, in the same order as defined during creation
            prediction = model.predict([image_input, measurements_input], verbose=0)[0]
            
            # --- Update State with New Prediction ---
            # This is the core of the stateful approach. The model's output becomes the next input.
            predicted_steering = float(prediction[0])
            predicted_throttle = float(prediction[1])
            
            if is_ai_enabled:
                # 4. Act in the Game based on the new state
                error = current_steering - predicted_steering
                print(f"AI ON | Predicted Steering: {predicted_steering:.2f}, Predicted Throttle: {predicted_throttle:.2f} | Current Steering: {current_steering:.2f}, Current Throttle: {current_throttle:.2f} | Error: {error:.2f}", end='\r')
                
                # Throttle control
                if predicted_throttle > 0.5:
                    keyboard.press('w')
                    keyboard.release('s')
                    current_throttle = 1.0
                elif predicted_throttle < -0.5:
                    keyboard.press('s')
                    keyboard.release('w')
                    current_throttle = -1.0
                else:
                    keyboard.release('w')
                    keyboard.release('s')
                    current_throttle = 0.0
                    
                # Steering control
                if (predicted_steering < -0.08 and (predicted_steering <= current_steering)): # Threshold for turning left  or (current_steering > 0.1 and (predicted_steering < current_steering * 0.8))
                    keyboard.press('a')
                    keyboard.release('d')
                    current_steering -= 0.1
                elif (predicted_steering > 0.08 and (predicted_steering >= current_steering)): # Threshold for turning right  or (current_steering < -0.1 and (predicted_steering > current_steering * 0.8))    
                    keyboard.press('d')
                    keyboard.release('a')
                    current_steering += 0.1
                else: # Drive straight
                    keyboard.release('a')
                    keyboard.release('d')
                    current_steering *= 0.95
                current_steering = np.clip(current_steering, -1.0, 1.0)
                # an internal state is kept with the same logic in the data collection script
            else:
                print(f"AI OFF | Predicted Steering: {predicted_steering:.2f}, Predicted Throttle: {predicted_throttle:.2f} | Current Steering: {current_steering:.2f}, Current Throttle: {current_throttle:.2f}", end='\r')
                if 'w' in keys_pressed:
                    current_throttle = 1.0
                elif 's' in keys_pressed:
                    current_throttle = -1.0
                else:
                    current_throttle = 0.0
                if 'a' in keys_pressed:
                    current_steering -= 0.1
                elif 'd' in keys_pressed:
                    current_steering += 0.1
                else:
                    current_steering *= 0.95
                current_steering = np.clip(current_steering, -1.0, 1.0)
                
                

                

    print("\nStopping AI.")
    listener.stop()
    # Release all keys as a safety measure
    keyboard.release('w')
    keyboard.release('a')
    keyboard.release('d')
    keyboard.release('s')

if __name__ == '__main__':
    main_loop()
