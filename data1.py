import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, regularizers

# ===============================
# 1. Kết nối MongoDB và lấy dữ liệu
# ===============================
client = MongoClient("mongodb+srv://NgocHan:ngochan6263@sweetdb.coon2qf.mongodb.net/SweetDB?retryWrites=true&w=majority&appName=SweetDB")
db = client["SweetDB"]
collection = db["DONHANG"]

orders = list(collection.find())

# Flatten items
records = []
for order in orders:
    user_id = str(order.get("user"))
    items = order.get("items", [])
    for item in items:
        product_id = str(item.get("productId"))
        quantity = item.get("quantity", 1)
        records.append({
            "user_id": user_id,
            "item_id": product_id,
            "rating": quantity
        })

df = pd.DataFrame(records)

# Pivot thành ma trận user-item
user_item_matrix = df.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# ===============================
# 1. Tạo Dataset và huấn luyện mô hình SVD
# ===============================
# Tạo Reader để xác định thang điểm đánh giá
reader = Reader(rating_scale=(1, 5))

# Tạo Dataset từ DataFrame
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
trainset, testset = train_test_split(data, test_size=0.2)

# Tạo mô hình SVD
model = SVD()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(trainset)

# ===============================
# 2. Dự đoán tất cả các sản phẩm cho người dùng và gợi ý top N sản phẩm
# ===============================
def recommend_top_n_products(model, user_id, top_n=5):
    """
    Gợi ý top N sản phẩm cho người dùng dựa trên mô hình SVD.
    """
    # Lấy tất cả sản phẩm mà người dùng chưa đánh giá
    all_item_ids = df['item_id'].unique()
    rated_items = df[df['user_id'] == user_id]['item_id'].tolist()
    unrated_items = [item for item in all_item_ids if item not in rated_items]
    
    # Dự đoán đánh giá cho tất cả các sản phẩm chưa đánh giá
    predictions = [model.predict(user_id, item) for item in unrated_items]
    
    # Sắp xếp các dự đoán theo đánh giá dự đoán (từ cao đến thấp)
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Chọn top N sản phẩm với đánh giá cao nhất
    top_n_products = [pred.iid for pred in predictions[:top_n]]
    
    return top_n_products, predictions

# ===============================
# 3. Tính RMSE và Precision@5
# ===============================
def compute_rmse_and_precision(model, testset, top_n=5):
    # Tính RMSE
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    
    # Tính Precision@5
    precisions = []
    for user_id, _, true_rating, est_rating, _ in predictions:
        # Nếu dự đoán được xếp vào top 5 của sản phẩm mà người dùng đã đánh giá
        top_n_products, _ = recommend_top_n_products(model, user_id, top_n)
        if true_rating > 0:  # Chỉ tính khi người dùng thực sự đã đánh giá
            precisions.append(1 if true_rating in top_n_products else 0)
    
    precision_at_5 = np.mean(precisions)
    return rmse, precision_at_5

# ===============================
# 4. Gọi hàm tính RMSE và Precision@5
# ===============================
rmse, precision_at_5 = compute_rmse_and_precision(model, testset, top_n=5)

# In kết quả
print(f"Test RMSE: {rmse:.4f}")
print(f"Precision@5: {precision_at_5:.4f}")
