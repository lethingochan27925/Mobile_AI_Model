from tensorflow.keras.models import load_model
import joblib

# Load model mà không compile để tránh lỗi
autoencoder = load_model("autoencoder_model.h5", compile=False)
encoder = load_model("encoder_model.h5", compile=False)

# Load scaler và danh sách product columns
scaler = joblib.load("scaler.pkl")
product_columns = joblib.load("product_columns.pkl")

