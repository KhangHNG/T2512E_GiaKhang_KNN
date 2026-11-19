import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# Bước 1. Nạp và khám phá dữ liệu
digits = load_digits()
X = digits.data        # 1.797 dòng × 64 cột
y = digits.target      # nhãn 0–9
anh = digits.images

# print(f"{X.shape} {y.shape} {anh.shape}")

# chu_so = np.bincount(y)
# for so, sl in enumerate(chu_so):
#     print(f"{so} - {sl}")

# print(f"Giá trị nhỏ nhất: {X.min()}")
# print(f"Giá trị lớn nhất: {X.max()}")

# fig, axes = plt.subplots(2, 10, figsize=(12, 3))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(anh[i], cmap="gray_r")
#     ax.set_title(y[i])
#     ax.axis("off")
# plt.show()

# Bước 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"số mẫu tập huấn luyện: {len(X_train)}")
print(f"số mẫu tập thử: {len(X_test)}\n")

if len(np.unique(y_test)) == 10:
    print("Đủ 10 số\n")
else:
    print(f"Thiếu {10 - len(X_test)} số\n")

# Bước 3. Mô hình đầu tiên
base = KNeighborsClassifier(n_neighbors=5)
base.fit(X_train, y_train)

accuracy = base.score(X_test, y_test)
print(f"Độ chính xác: {accuracy*100:.2f}%")

du_doan = base.predict(X_test[:10])
for i in range(10):
    if y_test[i] == du_doan[i]:
        status = "Dự đoán đúng"
    else:
        status = "Dự đoán sai"
    print(f"Nhãn thật: {y_test[i]} |Dự đoán {du_doan[i]} - {status}")

# Bước 4. Chọn tham số bằng GridSearchCV
luoi_tham_so = {
    "knn__n_neighbors": [1, 3, 5, 7, 9, 15, 25, 51],
    "knn__weights": ["uniform", "distance"],
}

pipeline = Pipeline([
    ("chuan_hoa", StandardScaler()),
    ("knn", KNeighborsClassifier()),     # tên bước: "knn"
])

tim_kiem = GridSearchCV(
    pipeline, luoi_tham_so,
    cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy"
)

tim_kiem.fit(X_train, y_train)

print(f"\nCấu hình tốt nhất: {tim_kiem.best_params_} - Điểm: {tim_kiem.best_score_}\n")

# ket_qua = pd.DataFrame(tim_kiem.cv_results_)
# plt.figure(figsize=(8, 5))
# for cach in ["uniform", "distance"]:
#     dong = ket_qua[ket_qua["param_knn__weights"] == cach]
#     plt.plot(dong["param_knn__n_neighbors"], dong["mean_test_score"], marker="o", label=cach)
# plt.xlabel("k"); plt.ylabel("Độ chính xác trung bình"); plt.legend()
# plt.show()

# Bước 5. Đánh giá trên tập kiểm tra
mo_hinh_tot_nhat = tim_kiem.best_estimator_
du_doan = mo_hinh_tot_nhat.predict(X_test)

acc_test = accuracy_score(y_test, du_doan)
so_sai = np.sum(du_doan != y_test)
print(f"Độ chính xác: {acc_test*100:2f}%")
print(f"Số ảnh đọc sai: {so_sai}\n")

# fig, ax = plt.subplots(figsize=(7, 7))
# ConfusionMatrixDisplay.from_predictions(y_test, du_doan, cmap="Blues", ax=ax)
# plt.title("Confusion Matrix")
# plt.show()

for so in range(10):
    vi_tri = (y_test == so)
    print(so, (du_doan[vi_tri] == so).mean())

# Bước 6. Phân tích lỗi bằng láng giềng
vi_tri_sai = np.where(du_doan != y_test)[0]
mo_hinh = tim_kiem.best_estimator_
k = mo_hinh.named_steps["knn"].n_neighbors

X_test_sai_chuan_hoa = mo_hinh.named_steps["chuan_hoa"].transform(X_test[vi_tri_sai])
khoang_cach, lang_gieng = mo_hinh.named_steps["knn"].kneighbors(X_test_sai_chuan_hoa)
fig, axes = plt.subplots(len(vi_tri_sai), k + 1, figsize=(2 * (k + 1), 2 * len(vi_tri_sai)))
for hang, i in enumerate(vi_tri_sai):
    axes[hang, 0].imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    axes[hang, 0].set_title(f"Thật: {y_test[i]} / Đoán: {du_doan[i]}", color="red")
    for cot in range(k):
        j = lang_gieng[hang, cot]
        axes[hang, cot + 1].imshow(X_train[j].reshape(8, 8), cmap="gray_r")
        axes[hang, cot + 1].set_title(f"{y_train[j]} ({khoang_cach[hang, cot]:.1f})")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

# Bước 7. Thử nghiệm với ảnh tự tạo
anh_goc = X_test[0].reshape(8, 8)
anh_dich = np.roll(anh_goc, shift=1, axis=1)   # dịch sang phải 1 cột
anh_dich[:, 0] = 0                             # xóa cột bị cuộn vòng sang trái

pred_goc = mo_hinh.predict(anh_goc.reshape(1, -1))[0]
pred_dich = mo_hinh.predict(anh_dich.reshape(1, -1))[0]

fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
axes[0].imshow(anh_goc, cmap="gray_r")
axes[0].set_title(f"Ảnh gốc | Đoán: {pred_goc}")
axes[0].axis("off")

axes[1].imshow(anh_dich, cmap="gray_r")
axes[1].set_title(f"Dịch phải 1px | Đoán: {pred_dich}")
axes[1].axis("off")

plt.tight_layout()
plt.show()

so_doi_ket_qua = 0
print("Kết quả kiểm tra dịch sang phải 1 điểm ảnh trên 10 ảnh đầu:")
for idx in range(10):
  g = X_test[idx].reshape(8, 8)
  d = np.roll(g, shift=1, axis=1)
  d[:, 0] = 0

  p_g = mo_hinh.predict(g.reshape(1, -1))[0]
  p_d = mo_hinh.predict(d.reshape(1, -1))[0]
  doi = p_g != p_d
  if doi:
    so_doi_ket_qua += 1
  print(f"Ảnh {idx:2d}: Gốc={p_g} -> Dịch={p_d} | Đổi dự đoán: {doi}")
  print(f"\nSố ảnh bị đổi kết quả sau khi dịch: {so_doi_ket_qua} / 10")

# Bước 8. Kết luận
  ket_luan = """
KNN tối ưu qua GridSearchCV đạt độ chính xác cao trên tập kiểm tra ảnh 8x8.
Lỗi chủ yếu rơi vào các chữ số có nét giống nhau khi ở độ phân giải thấp.
Do khoảng cách điểm ảnh nhạy cảm với độ lệch và độ nghiêng, dùng thực tế cần chuẩn hóa ảnh và gắn cờ các ca ít phiếu bầu hoặc khoảng cách xa để kiểm tra lại.
"""
print(ket_luan.strip())