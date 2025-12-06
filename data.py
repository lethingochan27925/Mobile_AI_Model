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

# ===============================
# 2. Tạo train/test split
# ===============================
X = user_item_matrix.copy()
train_matrix = X.copy()
test_matrix = pd.DataFrame(0, index=X.index, columns=X.columns)

for user in X.index:
    items = X.loc[user][X.loc[user] > 0].index.tolist()
    if len(items) < 2:
        train_items = items
        test_items = []
    else:
        train_items, test_items = train_test_split(items, test_size=0.2, random_state=42)
    train_matrix.loc[user, test_items] = 0
    test_matrix.loc[user, test_items] = X.loc[user, test_items]

train_values = train_matrix.values

# ===============================
# 3. Định nghĩa Autoencoder
# ===============================
input_layer = layers.Input(shape=(train_values.shape[1],))
# encoded = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
# encoded = layers.Dropout(0.4)(encoded)
# encoded = layers.Dense(64, activation='relu')(encoded)
# decoded = layers.Dense(128, activation='relu')(encoded)
# decoded = layers.Dense(train_values.shape[1], activation='sigmoid')(decoded)

encoded = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
encoded = layers.Dropout(0.4)(encoded)
encoded = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(encoded)

decoded = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(encoded)
decoded = layers.Dropout(0.3)(decoded)
decoded = layers.Dense(train_values.shape[1], activation='sigmoid')(decoded)


autoencoder = models.Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

import tensorflow as tf

mask = train_values > 0
mask = mask.astype(np.float32)

def masked_mse(y_true, y_pred):
    # mask = 1 nếu y_true > 0, else 0
    mask = tf.cast(tf.math.greater(y_true, 0), tf.float32)
    # tính MSE chỉ trên các rating >0
    mse = tf.square((y_true - y_pred) * mask)
    return tf.reduce_sum(mse) / tf.reduce_sum(mask)


autoencoder.compile(optimizer='adam', loss=masked_mse)




# ===============================
# 4. Huấn luyện Autoencoder
# ===============================

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
autoencoder.fit(train_values, train_values,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_split=0.1,
                callbacks=[early_stop])

# ===============================
# 5. Dự đoán ma trận test
# ===============================
predicted_test = autoencoder.predict(train_values)
predicted_test_matrix = pd.DataFrame(predicted_test, columns=X.columns, index=X.index)

# ===============================
# 6. Đánh giá model
# ===============================
mask = test_matrix.values > 0
rmse = np.sqrt(mean_squared_error(test_matrix.values[mask], predicted_test_matrix.values[mask]))
print(f"Test RMSE: {rmse:.4f}")

def precision_at_k(test_matrix, predicted_matrix, k=5):
    precisions = []
    for user in test_matrix.index:
        true_items = test_matrix.loc[user][test_matrix.loc[user] > 0].index.tolist()
        if len(true_items) == 0:
            continue
        pred_items = predicted_matrix.loc[user].sort_values(ascending=False).head(k).index.tolist()
        precisions.append(len(set(pred_items) & set(true_items)) / k)
    return np.mean(precisions)

precision = precision_at_k(test_matrix, predicted_test_matrix, k=5)
print(f"Precision@5: {precision:.4f}")

# ===============================
# 7. Hàm gợi ý cho Flask
# ===============================
def recommend_products(user_id, top_n=5):
    if user_id not in predicted_test_matrix.index:
        return []
    unrated_items = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0].index
    pred_ratings = predicted_test_matrix.loc[user_id, unrated_items]
    recommended_items = pred_ratings.sort_values(ascending=False).head(top_n).index.tolist()
    return recommended_items
