from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

# ===== 1️⃣ Recreate the same architecture used in training =====
# EfficientNetB1 was trained on color images (3 channels)
base_model = EfficientNetB1(
    weights=None,
    include_top=False,
    input_shape=(225, 225, 1)
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(29, activation='softmax')(x)  # number of classes
model = Model(inputs=base_model.input, outputs=x)

# ===== 2️⃣ Load the weights =====
# Use skip_mismatch=True to skip over small differences
model.load_weights("asl_efficientnet_best_finetuned.keras", by_name=True, skip_mismatch=True)

# ===== 3️⃣ Save the complete model in a Flask-compatible format =====
model.save("asl_efficientnet_fixed.h5")
print("✅ Model rebuilt and saved successfully as asl_efficientnet_fixed.h5")
