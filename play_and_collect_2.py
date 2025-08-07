# play_and_collect.py

# 1. IMPORTS
import time
import os
import numpy as np
import csv
from pynput.keyboard import Key, Controller, Listener
from PIL import Image
from collections import deque
from keras.models import load_model
import win32gui
import win32ui
import win32con
import ctypes
import threading

from keras import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense

# 2. CONFIGURATION & SETUP
# --- File Paths ---
MODEL_PATH = "./ats_driver_model_v2.h5"  # Path to your trained model
DATA_DIR = r"\\wsl.localhost\Ubuntu\home\raymi\projects\TruckDriver\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LOG_FILE = os.path.join(DATA_DIR, "log.csv")
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Model & Game ---
FRAME_STACK_SIZE = 4
WINDOW_TITLE = "American Truck Simulator"

# --- Script State & Thread Safety ---
# This dictionary will hold the global state of the script
STATE = {
    "running": True,
    "ai_enabled": False,
    "keys_pressed": set()
}
# --- FIX: Add a lock to prevent race conditions between the listener and main loop ---
KEYBOARD_LOCK = threading.Lock()
# ------------------------------------------------------------------------------------

# 3. UTILITY FUNCTIONS (from your original scripts)

# Make the script DPI-aware for accurate screen capture
ctypes.windll.user32.SetProcessDPIAware()

def capture_window_alt(hwnd):
    """Captures a window's contents using the PrintWindow API."""
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
    """Preprocesses a PIL Image and returns a NumPy array for the model."""
    left, top, right, bottom = 379, 222, 1293, 530
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((200, 66), Image.Resampling.LANCZOS)
    grayscale_img = resized_img.convert('L')
    image_array = np.array(grayscale_img, dtype=np.float32)
    normalized_array = image_array / 255.0
    return normalized_array[..., np.newaxis]

# 4. KEYBOARD HANDLING & STATE MANAGEMENT

def on_press(key):
    """Handles key presses for toggling state and recording user input."""
    global STATE
    # --- Kill Switch ---
    if key == Key.esc:
        print("\nStopping script...")
        STATE["running"] = False
        return False  # Stop the listener thread

    # --- Toggle AI / Intervention Mode ---
    if hasattr(key, 'char') and key.char == 'p':
        # --- FIX: Use the lock to ensure thread-safe state change ---
        with KEYBOARD_LOCK:
            STATE["ai_enabled"] = not STATE["ai_enabled"]
            mode = "AI DRIVING" if STATE["ai_enabled"] else "USER INTERVENING (RECORDING)"
            print(f"\n--- SWITCHING MODE TO: {mode} ---")
            
            # Release all keys that the AI might have been pressing
            release_all_keys()
            
            # Clear our own key state to prevent carrying over presses
            STATE["keys_pressed"].clear()
        # -------------------------------------------------------------
        return

    # --- Add pressed key to our set for data collection ---
    try:
        STATE["keys_pressed"].add(key.char)
    except AttributeError:
        pass  # Ignore special keys

def on_release(key):
    """Handles key releases for quitting and recording user input."""
    global STATE
    # --- Remove released key from our set ---
    try:
        STATE["keys_pressed"].remove(key.char)
    except (AttributeError, KeyError):
        pass

def release_all_keys():
    """Safety function to release all game control keys."""
    keyboard_controller = Controller()
    for key in ['w', 'a', 's', 'd']:
        keyboard_controller.release(key)

# 5. MAIN APPLICATION LOGIC

def main():
    """Main function to run the AI, handle interventions, and collect data."""
    print("--- Starting Self-Driving AI with Intervention System ---")
    
    # --- Initialization ---
    keyboard_controller = Controller()
    
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
    if not hwnd:
        print(f"ERROR: Game window '{WINDOW_TITLE}' not found!")
        STATE["running"] = False
        return

    try:
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
    except Exception as e:
        print(f"ERROR: Could not load model. {e}")
        STATE["running"] = False
        return

    frame_stack = deque(maxlen=FRAME_STACK_SIZE)

    log_header = ['timestamp', 'image_filename', 'steering_angle', 'throttle']
    log_file_exists = os.path.exists(LOG_FILE)
    log_f = open(LOG_FILE, 'a', newline='')
    writer = csv.writer(log_f)
    if not log_file_exists:
        writer.writerow(log_header)

    print("\nControls:")
    print("- [P] : Toggle between AI Driving and User Intervention/Recording.")
    print("- [Esc] : Quit the script.")
    print("\nStarting in 10 seconds... Click on the game window!")
    time.sleep(10)

    steering_angle = 0.0
    while STATE["running"]:
        loop_start_time = time.time()

        screen_raw = capture_window_alt(hwnd)
        if screen_raw is None: continue
        
        processed_frame = preprocess_image(screen_raw)
        frame_stack.append(processed_frame)

        if len(frame_stack) < FRAME_STACK_SIZE: continue
            
        # --- FIX: Acquire lock before letting AI control the keyboard ---
        with KEYBOARD_LOCK:
            if STATE["ai_enabled"]:
                # --- AI DRIVING MODE ---
                model_input = np.concatenate(list(frame_stack), axis=-1)
                model_input = np.expand_dims(model_input, axis=0)
                
                prediction = model.predict(model_input, verbose=0)[0]
                ai_steering, ai_throttle = prediction[0], prediction[1]
                
                print(f"\rAI Driving -> Steering: {ai_steering:.2f}, Throttle: {ai_throttle:.2f}", end="")

                if ai_throttle > 0.6: keyboard_controller.press('w')
                else: keyboard_controller.release('w')

                if ai_steering < -0.15:
                    keyboard_controller.press('a'); keyboard_controller.release('d')
                elif ai_steering > 0.15:
                    keyboard_controller.press('d'); keyboard_controller.release('a')
                else:
                    keyboard_controller.release('a'); keyboard_controller.release('d')
        # -----------------------------------------------------------------
        
        # This part runs outside the lock, as user input is direct
        if not STATE["ai_enabled"]:
            # --- USER INTERVENTION & RECORDING MODE ---
            timestamp = time.time()
            
            throttle = 1.0 if 'w' in STATE["keys_pressed"] else 0.0

            if 'a' in STATE["keys_pressed"]: steering_angle -= 0.1
            elif 'd' in STATE["keys_pressed"]: steering_angle += 0.1
            else: steering_angle *= 0.95
            steering_angle = np.clip(steering_angle, -1.0, 1.0)
            
            print(f"\rUser Intervening (RECORDING) -> Steering: {steering_angle:.2f}, Throttle: {throttle:.2f}", end="")

            image_filename = f"{timestamp}.npy"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            np.save(image_path, processed_frame)
            
            writer.writerow([timestamp, image_filename, steering_angle, throttle])

        time.sleep(max(1./30 - (time.time() - loop_start_time), 0))

    # --- Cleanup ---
    print("\nCleaning up...")
    release_all_keys()
    listener.stop()
    log_f.close()
    print("Script finished.")

if __name__ == '__main__':
    main()
