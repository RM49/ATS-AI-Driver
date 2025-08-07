# train.py (Stateful & Fixed)

# 1. IMPORTS
import pandas as pd
import numpy as np
import os
import tensorflow as tf # Import TensorFlow

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from keras.optimizers import Adam

def data_generator(log_df, batch_size=32, num_frames=4, augment=False):
    """
    A generator that yields batches of data for training.
    This version is FIXED to handle sequential data correctly.
    """
    # Create a list of all possible start indices for a full stack of frames.
    # We use the original, unshuffled log_df for this.
    valid_indices = [i for i in range(num_frames - 1, len(log_df))]
    
    while True:
        # Shuffle the list of valid start indices at the beginning of each epoch.
        np.random.shuffle(valid_indices)
        
        for i in range(0, len(valid_indices), batch_size):
            # Get a batch of starting indices
            batch_indices = valid_indices[i:i+batch_size]
            
            X_images_batch, X_measurements_batch, y_batch = [], [], []

            # Process each index in the batch
            for index in batch_indices:
                # --- Frame Stacking (Corrected) ---
                # This now correctly gets a sequence of frames from the unshuffled log
                stacked_frames = []
                for frame_offset in range(num_frames):
                    target_index = index - frame_offset
                    frame_log = log_df.iloc[target_index]
                    
                    # Assuming file paths are correct, this logic is now sound
                    image_path = os.path.join("data/images", frame_log['image_filename'])
                    frame = np.load(image_path)
                    stacked_frames.append(frame)

                # The frames were collected from newest to oldest, so reverse them for the stack
                X_images = np.concatenate(list(reversed(stacked_frames)), axis=-1)
                
                # --- Previous Measurements (Corrected) ---
                # Get the measurements from the frame immediately preceding the stack's start
                prev_log = log_df.iloc[index - 1]
                X_measurements = np.array([prev_log['steering_angle'], prev_log['throttle']], dtype=np.float32)

                # --- Target Label ---
                # The target is the action taken at the time of the newest frame
                current_log = log_df.iloc[index]
                y = np.array([current_log['steering_angle'], current_log['throttle']], dtype=np.float32)
                
                # --- Augmentation (Remains the same) ---
                if augment and np.random.rand() > 0.5:
                    X_images = np.fliplr(X_images)
                    X_measurements[0] = -X_measurements[0]
                    y[0] = -y[0]

                X_images_batch.append(X_images)
                X_measurements_batch.append(X_measurements)
                y_batch.append(y)
            
            yield (np.array(X_images_batch), np.array(X_measurements_batch)), np.array(y_batch)

# 3. LOAD AND SPLIT THE DATA LOG
log_df = pd.read_csv("data/log.csv")
log_df.dropna(inplace=True)
train_df, validation_df = train_test_split(log_df, test_size=0.2, random_state=42)

# 4. DEFINE THE MULTI-INPUT MODEL ARCHITECTURE (No changes here)
image_input = Input(shape=(66, 200, 4), name='image_input')
x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(image_input)
x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
image_features = Flatten()(x)

measurements_input = Input(shape=(2,), name='measurements_input')

combined = Concatenate()([image_features, measurements_input])

y = Dense(100, activation='relu')(combined)
y = Dense(50, activation='relu')(y)
y = Dense(10, activation='relu')(y)
output = Dense(2, name='output')(y)

model = Model(inputs=[image_input, measurements_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
model.summary()

# 5. CREATE TF.DATA DATASETS (THE FIX)
BATCH_SIZE = 32

# Define the data types and shapes for the generator's output
output_signature = (
    (tf.TensorSpec(shape=(None, 66, 200, 4), dtype=tf.float32),  # Input 1: Image stack
     tf.TensorSpec(shape=(None, 2), dtype=tf.float32)),         # Input 2: Measurements
    tf.TensorSpec(shape=(None, 2), dtype=tf.float32)             # Output: Target labels
)

# --- Create the training dataset ---
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_df, batch_size=BATCH_SIZE, augment=True),
    output_signature=output_signature
)

# --- Create the validation dataset ---
validation_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(validation_df, batch_size=BATCH_SIZE, augment=False),
    output_signature=output_signature
)

# 6. TRAIN THE MODEL
steps_per_epoch = max(1, len(train_df) // BATCH_SIZE)
validation_steps = max(1, len(validation_df) // BATCH_SIZE)

# Pass the tf.data.Dataset objects directly to model.fit
history = model.fit(
    train_dataset,
    epochs=35,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps
)

# 7. SAVE THE TRAINED MODEL
model.save("ats_driver_model_stateful.h5")
print("Stateful model saved successfully!")
