import tensorflow as tf
from tensorflow import keras
from keras.applications import (
    Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, 
    ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, 
    DenseNet121, DenseNet169, DenseNet201, NASNetMobile, NASNetLarge, 
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, 
    EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetV2B0, EfficientNetV2B1, 
    EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L,
    ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
)

# Dictionary to store models
models = {}

# List of model classes
model_classes = [
    Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, 
    ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, 
    DenseNet121, DenseNet169, DenseNet201, NASNetMobile, NASNetLarge, 
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, 
    EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetV2B0, EfficientNetV2B1, 
    EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L,
    ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
]

# Load each model
for model_class in model_classes:
    model_name = model_class.__name__
    try:
        models[model_name] = model_class(weights='imagenet')
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Example: Access a specific model
# print(models['MobileNet'].summary())
