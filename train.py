# train.py

# 1. IMPORTS
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.optimizers import Adam

def data_generator(log_df, batch_size=32, num_frames=4, augment=False):
    while True:
        log_df = log_df.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(log_df), batch_size):
            batch_logs = log_df.iloc[i:i+batch_size]
            X_batch, y_batch = [], []
            for index, row in batch_logs.iterrows():
                # --- START MODIFICATION ---
                
                stacked_frames = []
                sample_is_valid = True 

                try:
                    for frame_index in range(num_frames):
                        target_index = max(0, index - frame_index) 
                        frame_log = log_df.iloc[target_index]
                        
                        image_path = os.path.join("data/images", frame_log['image_filename'])
                        frame = np.load(image_path)
                        stacked_frames.append(frame)
                        
                except FileNotFoundError:
                    print(f"\nWarning: Missing file for log entry at index {index}. Skipping sample.")
                    sample_is_valid = False
                
                if sample_is_valid:
                    X = np.concatenate(list(reversed(stacked_frames)), axis=-1)
                    y = np.array([row['steering_angle'], row['throttle']], dtype=np.float32)

                    # --- AUGMENTATION LOGIC ---
                    if augment and np.random.rand() > 0.5:
                        # Flip the image horizontally
                        X = np.fliplr(X)
                        # Invert the steering angle
                        y[0] = -y[0]
                    # --- END AUGMENTATION ---

                    X_batch.append(X)
                    y_batch.append(y)
                
            if X_batch:
                yield np.array(X_batch), np.array(y_batch)

# 3. LOAD AND SPLIT THE DATA LOG
log_df = pd.read_csv("data/log.csv")
train_df, validation_df = train_test_split(log_df, test_size=0.2)

# 4. DEFINE YOUR UPDATED MODEL ARCHITECTURE
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

#model.load_weights("./ats_driver_model_v2.h5")

# The loss function 'mean_squared_error' works perfectly for multiple outputs.
# It will calculate the error for steering and throttle and average them.
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
model.summary()

# 5. CREATE GENERATOR INSTANCES
BATCH_SIZE = 32
# Activate augmentation for the training generator
train_gen = data_generator(train_df, batch_size=BATCH_SIZE, augment=True) 
val_gen = data_generator(validation_df, batch_size=BATCH_SIZE, augment=False) # No need to augment validation data

# 6. TRAIN THE MODEL
history = model.fit(
    train_gen,
    epochs=60,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=len(validation_df) // BATCH_SIZE
)

# 7. SAVE THE TRAINED MODEL
model.save("ats_driver_model_v2.h5")
print("Model saved!")
