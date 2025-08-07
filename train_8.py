# train.py (Stateful & 6x Deterministic Augmentation - 8 Frame Variant)

# 1. IMPORTS
import pandas as pd
import numpy as np
import os
import cv2 # Import OpenCV for image processing
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from keras.optimizers import Adam

# --- MODIFIED: Default number of frames changed to 8 ---
def data_generator(log_df, batch_size=32, num_frames=8, augment=False):
    """
    A generator that yields batches of data.
    If augment=True, it sextuples the training data by deterministically creating
    flipped, blurred, sharpened, and combinations thereof for each original stack.
    If augment=False, it yields single, original samples for validation.
    """
    # This ensures we have enough preceding frames to create a full stack
    valid_indices = [i for i in range(num_frames - 1, len(log_df))]

    # Define the sharpening kernel once to avoid re-creation
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    while True:
        np.random.shuffle(valid_indices)

        # Batch accumulation lists
        X_images_batch, X_measurements_batch, y_batch = [], [], []

        for index in valid_indices:
            # --- Base Sample Generation (common logic) ---
            stacked_frames = []
            # Collect the sequence of frames leading up to the current index
            for frame_offset in range(num_frames):
                target_index = index - frame_offset
                frame_log = log_df.iloc[target_index]
                image_path = os.path.join("data/images", frame_log['image_filename'])
                frame = np.load(image_path)
                stacked_frames.append(frame)

            # Concatenate frames along the channel axis. The list is reversed to maintain chronological order (oldest to newest).
            base_X_images = np.concatenate(list(reversed(stacked_frames)), axis=-1)

            # Get the measurements from the previous timestep as input
            prev_log = log_df.iloc[index - 1]
            base_X_measurements = np.array([prev_log['steering_angle'], prev_log['throttle']], dtype=np.float32)

            # Get the measurements from the current timestep as the target label
            current_log = log_df.iloc[index]
            base_y = np.array([current_log['steering_angle'], current_log['throttle']], dtype=np.float32)

            # --- Augmentation & Batching Logic ---
            if not augment:
                # For validation, just process the single, original sample
                items_to_process = [(base_X_images, base_X_measurements, base_y)]
            else:
                # For training, create all 6 versions to sextuple the data

                # Version 1: Original
                item1 = (base_X_images, base_X_measurements, base_y)

                # Version 2: Flipped
                flipped_X_images = np.fliplr(base_X_images)
                flipped_X_measurements = np.copy(base_X_measurements)
                flipped_X_measurements[0] *= -1 # Invert steering angle
                flipped_y = np.copy(base_y)
                flipped_y[0] *= -1 # Invert target steering angle
                item2 = (flipped_X_images, flipped_X_measurements, flipped_y)

                # Version 3: Blurred
                blurred_X_images = np.copy(base_X_images)
                for i in range(num_frames):
                    # Apply Gaussian Blur to each frame in the stack
                    blurred_X_images[:, :, i] = cv2.GaussianBlur(base_X_images[:, :, i], (3, 3), 0)
                item3 = (blurred_X_images, base_X_measurements, base_y)

                # Version 4: Flipped + Blurred
                flipped_blurred_X_images = np.copy(flipped_X_images)
                for i in range(num_frames):
                    flipped_blurred_X_images[:, :, i] = cv2.GaussianBlur(flipped_X_images[:, :, i], (3, 3), 0)
                item4 = (flipped_blurred_X_images, flipped_X_measurements, flipped_y)

                # Version 5: Sharpened
                sharpened_X_images = np.copy(base_X_images)
                for i in range(num_frames):
                    # Apply sharpening filter to each frame in the stack
                    sharpened_X_images[:, :, i] = cv2.filter2D(src=base_X_images[:, :, i], ddepth=-1, kernel=sharpen_kernel)
                item5 = (sharpened_X_images, base_X_measurements, base_y)

                # Version 6: Flipped + Sharpened
                flipped_sharpened_X_images = np.copy(flipped_X_images)
                for i in range(num_frames):
                    flipped_sharpened_X_images[:, :, i] = cv2.filter2D(src=flipped_X_images[:, :, i], ddepth=-1, kernel=sharpen_kernel)
                item6 = (flipped_sharpened_X_images, flipped_X_measurements, flipped_y)

                items_to_process = [item1, item2, item3, item4, item5, item6]

            # Add the generated item(s) to the batch and yield if full
            for img, meas, label in items_to_process:
                X_images_batch.append(img)
                X_measurements_batch.append(meas)
                y_batch.append(label)

                if len(X_images_batch) >= batch_size:
                    yield (np.array(X_images_batch), np.array(X_measurements_batch)), np.array(y_batch)
                    # Reset for the next batch
                    X_images_batch, X_measurements_batch, y_batch = [], [], []


# 3. LOAD AND SPLIT THE DATA LOG
log_df = pd.read_csv("data/log.csv")
log_df.dropna(inplace=True)
train_df, validation_df = train_test_split(log_df, test_size=0.2, random_state=42)

# 4. DEFINE THE MULTI-INPUT MODEL ARCHITECTURE
# --- MODIFIED: Input shape channel depth changed from 4 to 8 ---
image_input = Input(shape=(66, 200, 8), name='image_input')
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

# 5. CREATE TF.DATA DATASETS
BATCH_SIZE = 32
NUM_FRAMES = 8 # Define as a constant for clarity

# --- MODIFIED: TensorSpec shape updated to reflect 8 frames ---
output_signature = (
    (tf.TensorSpec(shape=(None, 66, 200, NUM_FRAMES), dtype=tf.float32),  # Input 1: Image stack
     tf.TensorSpec(shape=(None, 2), dtype=tf.float32)),         # Input 2: Measurements
    tf.TensorSpec(shape=(None, 2), dtype=tf.float32)             # Output: Target labels
)

# --- Create the training dataset ---
# --- MODIFIED: Explicitly pass num_frames=NUM_FRAMES ---
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_df, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, augment=True),
    output_signature=output_signature
)

# --- Create the validation dataset ---
# --- MODIFIED: Explicitly pass num_frames=NUM_FRAMES ---
validation_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(validation_df, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, augment=False),
    output_signature=output_signature
)

# 6. TRAIN THE MODEL
# With 6x augmentation, the number of training samples per epoch is sextupled.
steps_per_epoch = max(1, (len(train_df) * 6) // BATCH_SIZE)
validation_steps = max(1, len(validation_df) // BATCH_SIZE)

print(f"Training with {steps_per_epoch} steps per epoch using {NUM_FRAMES} frames.")

# Pass the tf.data.Dataset objects directly to model.fit
history = model.fit(
    train_dataset,
    epochs=25,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps
)

# 7. SAVE THE TRAINED MODEL
# --- MODIFIED: Filename updated to reflect 8-frame model ---
model.save("ats_driver_model_8_frame_6x_augmented.h5")
print("8-frame, 6x augmented model saved successfully!")
