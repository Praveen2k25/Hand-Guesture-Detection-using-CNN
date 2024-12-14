import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


data_dir = 'dataset'
classes = ['open hand', 'closed hand']



def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0 
    img_reshaped = np.reshape(img_normalized, (64, 64, 1))  
    return img_reshaped



def load_dataset(data_dir):
    images = []
    labels = []

    for idx, gesture in enumerate(classes):
        gesture_dir = os.path.join(data_dir, gesture)
        for img_name in os.listdir(gesture_dir):
            img_path = os.path.join(gesture_dir, img_name)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(idx)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels



def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(
        Dense(len(classes), activation='softmax'))  

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
 
    images, labels = load_dataset(data_dir)

   
    labels_categorical = to_categorical(labels, num_classes=len(classes))


    X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)


    model = build_model()


    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)


    model.save('hand_gesture_model_gray.h5')
    print("Model saved as hand_gesture_model_gray.h5")


if __name__ == '__main__':
    main()
