import numpy as np
import pandas as pd
from models import autoencoder, scaler  # import model + scaler
from pymongo import MongoClient
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv


load_dotenv()

# ------------------- BUILD USER-ITEM MATRIX -------------------
def build_user_item_matrix():
    """
    Gộp dữ liệu fake (Excel) + dữ liệu thật (MongoDB)
    Trả về user-item matrix dạng numpy array
    """
    # --- Fake data ---

    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


    fake_path = os.path.join(BASE_DIR, "customer.xlsx")
    
    fake_customers = pd.read_excel(fake_path).head(1000)
    fake_customers['ID'] = range(1, len(fake_customers)+1)

    # --- Lấy sản phẩm từ Mongo ---
    client = MongoClient(os.getenv("URLClient"))
    db = client["SweetDB"]
    products = list(db["SANPHAM"].find())
    product_ids = [str(p["_id"]) for p in products]

    # --- Fake orders ---

    # --- Fake orders từ Excel ---
    # file_path_orders = r'..\orders_27500.xlsx'
    file_path_orders = os.path.join(BASE_DIR, "orders_27500.xlsx")
    fake_orders_df = pd.read_excel(file_path_orders)

    # --- MongoDB thật ---

    real_orders = []
    for order in db["DONHANG"].find():
        for item in order.get("items", []):
            real_orders.append({
                'customer_id': str(order['user']),
                'product_id': str(item['productId']),
                'quantity': int(item['quantity'])
            })

    real_orders_df = pd.DataFrame(real_orders)

    # --- Gộp fake + thật ---
    all_orders_df = pd.concat([fake_orders_df, real_orders_df], ignore_index=True)

    # --- Pivot thành ma trận user-item ---
    user_item_matrix = all_orders_df.pivot_table(
        index='customer_id',
        columns='product_id',
        values='quantity',
        aggfunc='sum',
        fill_value=0
    )

    # --- Chuyển thành numpy ---
    X = user_item_matrix.values

    # --- Chuẩn hóa ---
    X_scaled = scaler.fit_transform(X) #scale toàn bộ 
    # X_scaled = scaler.transform(X)  # scale 1 phần

    return X_scaled

# ------------------- RETRAIN MODEL -------------------

tf.config.run_functions_eagerly(True)



def retrain_model():
    print("Bắt đầu retrain model...")
    
    # --- Build ma trận user-item ---
    X_scaled = build_user_item_matrix()

    # --- Load model cũ ---
    autoencoder = load_model("autoencoder_model.h5", compile=False)

    # --- Compile lại optimizer để tránh lỗi Unknown variable ---
    # autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    autoencoder.compile(optimizer='adam', loss='mse')


    # --- Train lại ---
    autoencoder.fit(X_scaled, X_scaled, epochs=5, batch_size=32, verbose=1)

    # --- Lưu model + scaler ---
    autoencoder.save("autoencoder_model.h5")
    joblib.dump(scaler, "scaler.pkl")
    
    print("Đã retrain và lưu model thành công!")
