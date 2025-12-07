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


# ==============================
#         HÀM HỖ TRỢ
# ==============================

def get_closest_name(rgb_centers, df):
    r, g, b = rgb_centers
    distance = ((df['R'] - r)**2 + (df['G'] - g)**2 + (df['B'] - b)**2) ** 0.5
    closest_row_index = distance.idxmin()
    return df.loc[closest_row_index, 'color_name']


def resize(np_img, basewidth=500):
    img = Image.fromarray(np_img)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * wpercent))
    img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return np.array(img)


def hexify(palette):
    return ['#%s' % ''.join(('%02x' % round(p) for p in colour)) for colour in palette]


# ==============================
#     TIỀN XỬ LÝ + KMEANS
# ==============================

def KMeansModel(imgfile=None, n_clusters=5):

    img = image.imread(imgfile)

    # PNG float (0–1)
    if img.dtype in [np.float32, np.float64]:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Xám → RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # RGBA → RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = resize(img, 500)

    rgb = img.reshape((-1, 3))
    hsv = colors.rgb_to_hsv(img / 255.0)
    hsv_flat = (hsv.reshape((-1, 3)) * 255)

    features = np.hstack((rgb, hsv_flat))

    cluster = KMeans(
        n_clusters=n_clusters,
        init='random',
        n_init=10,
        max_iter=300,
        random_state=0
    )
    cluster.fit_predict(features)

    return cluster, img


# ==============================
#     EXTRACT FEATURE CHO KNN
# ==============================

def extract_for_knn(cluster, img):

    hsv = colors.rgb_to_hsv(img / 255.0)
    mean_brightness = np.mean(hsv[:, :, 2]) * 255

    centers = cluster.cluster_centers_
    sort_index = np.argsort(centers[:, 5])
    sorted_centers = centers[sort_index]

    hsv_only = sorted_centers[:, 3:6]
    flat_features = hsv_only.flatten()

    return np.insert(flat_features, 0, mean_brightness)


# ==============================
#       VẼ KẾT QUẢ
# ==============================

def plot_results(cluster, img, df_colors, day_or_night):
    centers = cluster.cluster_centers_
    sort_index = np.argsort(centers[:, 5])
    sorted_centers = centers[sort_index]

    rgb_centers = np.int_(sorted_centers[:, 0:3].round())
    colors_labels = hexify(rgb_centers)

    name_labels = []
    if df_colors is not None:
        for rgb in rgb_centers:
            name_labels.append(get_closest_name(rgb, df_colors))
    else:
        name_labels = ["No CSV"] * len(colors_labels)

    labels = [f"{name}\n{h}" for name, h in zip(name_labels, colors_labels)]

    fig = plt.figure(figsize=(6, 6))
    gs = grd.GridSpec(2, 1, height_ratios=[4, 1])
    sns.set_style("whitegrid")

    # Ảnh input
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.axis("off")

    title_color = "orange" if "NGÀY" in day_or_night else "darkblue"
    ax1.set_title(day_or_night, fontsize=16,
                  fontweight="bold", color=title_color)

    # Palette
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(labels))
    ax2.bar(x, 1, width=1.0, color=rgb_centers / 255)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.close(fig)
    return fig


def colormap(cluster, img):
    colours = np.int_(cluster.cluster_centers_[:, 0:3].round())
    cmap = colours[cluster.labels_].reshape(img.shape[0], img.shape[1], 3)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cmap)
    plt.axis("off")
    plt.grid(False)

    plt.close(fig)
    return fig


# ==============================
#         HÀM CHÍNH
# ==============================

def run_prediction(image_path):
    csv_path = "color_names.csv"
    model_path = "day_night_knn_model.pkl"

    df_colors = None
    if os.path.exists(csv_path):
        df_colors = pd.read_csv(csv_path)
        df_colors.rename(columns={
            "Name": "color_name",
            "Red (8 bit)": "R",
            "Green (8 bit)": "G",
            "Blue (8 bit)": "B"
        }, inplace=True)
        df_colors = df_colors[["color_name", "R", "G", "B"]]

    if not os.path.exists(model_path):
        raise FileNotFoundError("Không tìm thấy model KNN!")

    knn_model = joblib.load(model_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError("Không tìm thấy file ảnh!")

    cluster_model, resized_img = KMeansModel(image_path, n_clusters=5)

    feature_vector = extract_for_knn(cluster_model, resized_img).reshape(1, -1)
    pred = knn_model.predict(feature_vector)

    result_label = "NGÀY (Day)" if pred[0] == 1 else "ĐÊM (Night)"

    fig1 = plot_results(cluster_model, resized_img, df_colors, result_label)
    fig2 = colormap(cluster_model, resized_img)

    return result_label, fig1, fig2


# ==============================
#       TEST LOCAL
# ==============================
if __name__ == "__main__":
    label, fig1, fig2 = run_prediction("anh-cua_minh.jpg")
    print(label)
    fig1.show()
    fig2.show()
