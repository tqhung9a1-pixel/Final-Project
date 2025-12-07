import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.ticker as ticker
from matplotlib import image, colors #thêm vào clors để tính hsv
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier #lôi knn từ sklearn ra
import seaborn as sns

def get_closest_name(rgb_centers, df):
    r, g, b = rgb_centers
    distance = ((df['R'] - r)**2 + (df['G'] - g)**2 + (df['B'] - b)**2) ** 0.5
    closest_row_index = distance.idxmin()
    return df.loc[closest_row_index, 'color_name']



def resize(np_img,basewidth=200):
    img = Image.fromarray(np_img)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return np.array(img)

def hexify(palette):
    return['#%s' % ''.join(('%02x' % round(p) for p in colour)) for colour in palette]

def KMeansModel(imgfile=None,n_clusters=5):

    img = image.imread(imgfile)
#---phòng trường hợp file png (0-1):
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img* 255).astype(np.uint8) 
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    img = resize(img,500) #resize img khi đưa vào Kmeans

    rgb = img.reshape((img.shape[0]*img.shape[1],3))
    hsv = colors.rgb_to_hsv(img / 255.0)
    hsv_flat = hsv.reshape((hsv.shape[0]*hsv.shape[1],3))*255

    features = np.hstack((rgb,hsv_flat))

    
    cluster = KMeans(n_clusters=n_clusters,init='random',n_init=10, max_iter=300,random_state=0)
    cluster.fit_predict(features)
    
    return cluster, img

def extract_for_knn(cluster, img): # --> thêm img


    # Thêm: Tính mean brightness
    hsv = colors.rgb_to_hsv(img / 255.0) # img này là resized_img truyền vào
    mean_brightness = np.mean(hsv[:, :, 2]) * 255
    # ------------------------------------


    centers_1 = cluster.cluster_centers_
    sort_index_again = np.argsort(centers_1[:, 5])
    sorted_centers_again = centers_1[sort_index_again] # sort lại lần nữa cho knn
    hsv_only = sorted_centers_again[:, 3:6] 
    flat_features = hsv_only.flatten()
    
    # return flat_features
    # Chèn thêm mean_brightness vào đầu ---
    return np.insert(flat_features, 0, mean_brightness)



def plot_results(cluster, img, df_colors, day_or_night):
    centers = cluster.cluster_centers_
    sort_index = np.argsort(centers[:, 5])
    sorted_centers = centers[sort_index] # thứ tự tăng dần (V)
    print(np.int_(sorted_centers.round()))
   
    rgb_centers = np.int_(sorted_centers[:, 0:3].round())
    
    
    colors_labels = hexify(rgb_centers)

    #Tìm cái name của màu
    name_lables = []
    if df_colors is not None:
        for rgb in rgb_centers:
            name = get_closest_name(rgb, df_colors)
            name_lables.append(name)
    else:
        name_lables = ['No CSV'] * len(colors_labels)

    labels = [f"{name}\n{h_code}" for name, h_code in zip(name_lables, colors_labels)]

    
    fig = plt.figure(figsize=(10,10))
    gs = grd.GridSpec(2, 1, height_ratios=[4,1])
    sns.set_style('whitegrid')
    
    
    
    # Image plot
    
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    plt.axis ("off")
    plt.grid(False)

    label_text = day_or_night
    title_color = "orange" if label_text == "NGÀY (Day)" else "darkblue"
    ax1.set_title(f"{label_text}" , fontsize=20, color=title_color, fontweight='bold')
    
    # Colour palette

    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(labels))
    y = np.ones(len(x))
    ax2.bar(x,y,width=1.0,color=rgb_centers/256)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels,rotation=20)
    
    ax2.get_yaxis().set_ticks([])

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.grid(False)
    

    plt.show()
    return labels

def colormap(cluster,img):
    colours = np.int_(cluster.cluster_centers_[:, 0:3].round())
    cmap = colours[cluster.labels_]
    cmap = cmap.reshape((img.shape[0],img.shape[1],3))
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cmap)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show()


if __name__=="__main__":
    image_path = r'484244206_937911511503343_437031781001401646_n.png'
    csv_path = 'day_night_images\color_names.csv'
    model_path = "day_night_knn_model.pkl"


    df_colors = None
    if os.path.exists(csv_path):
        try:
            df_colors = pd.read_csv(csv_path)

            rename_map = {
                "Name": "color_name",
                'Red (8 bit)': 'R',
                'Green (8 bit)': 'G',
                'Blue (8 bit)': 'B'

                }
            df_colors.rename(columns=rename_map, inplace=True)

            df_colors = df_colors[['color_name', 'R', 'G', 'B']]
        except Exception as e:
            print(f"Lỗi đọc CSV: {e}")
            df_colors = None
    

    if os.path.exists(model_path):
        knn_model = joblib.load(model_path)
    else:
        print(f">>> KHÔNG TÌM THẤY MODEL '{model_path}'")

    if os.path.exists(image_path):
        cluster_model, resized_img = KMeansModel(imgfile=image_path, n_clusters=5)


        # Thêm resized_img vào 
        feature_vector = extract_for_knn(cluster_model, resized_img) 
        pred = knn_model.predict(feature_vector.reshape(1, -1))
        # ------------------------------------------



        # pred = knn_model.predict(extract_for_knn(cluster_model).reshape(1, -1))
        
        
        result_label = "NGÀY (Day)" if pred[0] == 1 else "ĐÊM (Night)"
        print(result_label)

        labels = plot_results(cluster_model, resized_img, df_colors, result_label)
        colormap(cluster_model, resized_img)
    else:
        print("Không tìm thấy file ảnh!")
    
    
       
    
            
