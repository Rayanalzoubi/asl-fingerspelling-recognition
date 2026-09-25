from tensorflow.keras.models import load_model

model = load_model("asl_efficientnet_best_finetuned.keras", compile=False)
print("\n✅ Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
