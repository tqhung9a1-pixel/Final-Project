import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from matplotlib import image, colors
import os
import joblib
from sklearn.cluster import KMeans
import seaborn as sns

plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["figure.dpi"] = 120


def get_closest_name(rgb_centers, df):
    r, g, b = rgb_centers
    distance = ((df['R'] - r)**2 + (df['G'] - g)**2 + (df['B'] - b)**2) ** 0.5
    closest_row_index = distance.idxmin()
    return df.loc[closest_row_index, 'color_name']


def resize(np_img, basewidth=200):
    img = Image.fromarray(np_img)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return np.array(img)


def hexify(palette):
    return ['#%s' % ''.join(('%02x' % round(p) for p in colour)) for colour in palette]


def KMeansModel(imgfile=None, n_clusters=5):
    img = image.imread(imgfile)

    # Chuẩn hóa kiểu dữ liệu
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Đảm bảo 3 kênh RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)  # Grayscale → RGB
    elif img.shape[2] == 4:
        img = img[:, :, :3]  # RGBA → RGB

    img = resize(img, 200)

    # Trích xuất đặc trưng
    rgb = img.reshape((img.shape[0] * img.shape[1], 3))
    hsv = colors.rgb_to_hsv(img / 255.0)
    hsv_flat = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3)) * 255
    features = np.hstack((rgb, hsv_flat))

    cluster = KMeans(n_clusters=n_clusters, init='random',
                     n_init=10, max_iter=300, random_state=0)
    cluster.fit_predict(features)

    return cluster, img


def extract_for_knn(cluster):
    centers = cluster.cluster_centers_
    sort_index = np.argsort(centers[:, 5])
    sorted_centers = centers[sort_index]
    hsv_only = sorted_centers[:, 3:6]
    flat_features = hsv_only.flatten()
    return flat_features


def plot_results(cluster, img, df_colors, day_or_night):
    centers = cluster.cluster_centers_
    sort_index = np.argsort(centers[:, 5])
    sorted_centers = centers[sort_index]
    rgb_centers = np.int_(sorted_centers[:, 0:3].round())
    colors_labels = hexify(rgb_centers)

    name_labels = []
    if df_colors is not None:
        for rgb in rgb_centers:
            name = get_closest_name(rgb, df_colors)
            name_labels.append(name)
    else:
        name_labels = ['No CSV'] * len(colors_labels)

    labels = [f"{name}\n{h_code}" for name,
              h_code in zip(name_labels, colors_labels)]

    fig = plt.figure(figsize=(6, 6))  # ← tăng kích thước
    gs = grd.GridSpec(2, 1, height_ratios=[4, 1])
    sns.set_style('whitegrid')

    # Ảnh gốc
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.axis("off")
    title_color = "orange" if "NGÀY" in day_or_night else "darkblue"
    ax1.set_title(day_or_night, fontsize=15,
                  color=title_color, fontweight='bold')

    # Palette màu
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(labels))
    y = np.ones(len(x))
    ax2.bar(x, y, width=1.0, color=rgb_centers / 256)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(False)

    plt.close(fig)
    return fig


def colormap(cluster, img, show=False):
    colours = np.int_(cluster.cluster_centers_[:, 0:3].round())
    cmap = colours[cluster.labels_]
    cmap = cmap.reshape((img.shape[0], img.shape[1], 3))
    fig = plt.figure(figsize=(6, 6))  # ← tăng kích thước
    plt.imshow(cmap)
    plt.axis('off')
    plt.grid(False)

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig


# === HÀM CHÍNH DÙNG CHO STREAMLIT HOẶC SCRIPT ===
def run_prediction(image_path="anh-cua_minh.jpg"):
    csv_path = "color_names.csv"
    model_path = "day_night_knn_model.pkl"

    # Tải file màu (nếu có)
    df_colors = None
    if os.path.exists(csv_path):
        try:
            df_colors = pd.read_csv(csv_path)
            df_colors.rename(columns={
                "Name": "color_name",
                'Red (8 bit)': 'R',
                'Green (8 bit)': 'G',
                'Blue (8 bit)': 'B'
            }, inplace=True)
            df_colors = df_colors[['color_name', 'R', 'G', 'B']]
        except Exception as e:
            print(f"Lỗi đọc CSV: {e}")

    # Tải model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    knn_model = joblib.load(model_path)

    # Kiểm tra ảnh
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    # Xử lý
    cluster_model, resized_img = KMeansModel(imgfile=image_path, n_clusters=5)
    features = extract_for_knn(cluster_model).reshape(1, -1)
    pred = knn_model.predict(features)
    result_label = "NGÀY (Day)" if pred[0] == 1 else "ĐÊM (Night)"

    # Vẽ biểu đồ
    fig1 = plot_results(cluster_model, resized_img, df_colors, result_label)
    fig2 = colormap(cluster_model, resized_img, show=False)

    return result_label, fig1, fig2


# === CHẠY THỦ CÔNG ===
if __name__ == "__main__":
    try:
        label, fig1, fig2 = run_prediction()
        print(label)
        # Hiển thị biểu đồ khi chạy script
        plt.figure(fig1.number)
        plt.show()
        plt.figure(fig2.number)
        plt.show()
    except Exception as e:
        print(f"Lỗi: {e}")
