import warnings
warnings.filterwarnings("ignore")
import os
import gc
import shutil
import time
import glob
import random
import numpy as np 
import pandas as pd
import json
import cv2
from tqdm import tqdm
tqdm.pandas()
from scipy.io import loadmat


def convertBoxFormat(box):
    (box_x_left, box_y_top, box_w, box_h) = box
    (image_w, image_h) = (640, 480)
    dw = 1./image_w
    dh = 1./image_h
    x = (box_x_left + box_w / 2.0) * dw
    y = (box_y_top + box_h / 2.0) * dh
    w = box_w * dw
    h = box_h * dh
    return (x, y, w, h)
    

# https://github.com/simonzachau/caltech-pedestrian-dataset-to-yolo-format-converter
# Generate Images from Video Files
def save_img(dir_path, fn, i, frame):
    cv2.imwrite('{}/{}_{}_{}.png'.format(
        dir_path, fn.split('/')[-2], os.path.basename(fn).split('.')[0], f"{i:04d}"), 
        frame)
    

def convert_caltech(split, df):
    # Directory Path
    print(split)
    input_dir = "caltechpedestriandataset"
    output_dir = "datasets/caltechpedestriandataset/images"
    if(split=="Train"):
        output_dir = os.path.join(output_dir, "train")
    else:
        output_dir = os.path.join(output_dir, "val")

    if(os.path.exists(output_dir)==False):
        os.mkdir(output_dir)
    
    # Sets
    sets_list = sorted(glob.glob(os.path.join(input_dir, split+"/*")))
    print("Total Sets:", len(sets_list))
    for dname in sets_list:
        print(dname)
        dname2 = dname.split("/")[-1]
        df_filtered = df[df["set_id"]==dname2].reset_index(drop=True)
        
        # Videos
        videos_list = list(df_filtered["video_id"].unique())
        print("Total Videos:", len(videos_list))
        for i, vd in enumerate(videos_list):
            fn = os.path.join(dname, dname2, vd+".seq")
            print(fn)
            cap = cv2.VideoCapture(fn)
            df_filtered2 = df_filtered[df_filtered["video_id"]==vd]
            
            # Frames
            frame_set = set(df_filtered2["frame_id"].unique())
            limit = len(frame_set)
            print("Total Frames:", limit)
            j = 0
            k = 0
            while True:
                ret, frame = cap.read()
                if(j in frame_set):
                    save_img(output_dir, fn, j, frame)
                    k += 1
                    if(k==limit):
                        break
                j += 1
                

def generate_labels(split, df):
    # Directory Path
    output_dir = "datasets/caltechpedestriandataset/labels"
    output_dir = os.path.join(output_dir, split)
    if(os.path.exists(output_dir)==False):
        os.mkdir(output_dir)
        
    set_id_list = list(df["set_id"].unique())
    for set_id in set_id_list:
        df_set_id = df[df["set_id"]==set_id].reset_index(drop=True)
        for idx, row in df_set_id.iterrows():
            label_file = open(output_dir + "/" + row["image_id"] + ".txt", 'w')
            label_file.write(row["label"])
            label_file.close()
            

if __name__ == "__main__":

    os.makedirs(os.path.join(os.path.dirname(__file__),
                             "datasets/caltechpedestriandataset/images/train"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__),
                             "datasets/caltechpedestriandataset/images/val"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__),
                             "datasets/caltechpedestriandataset/labels/train"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__),
                             "datasets/caltechpedestriandataset/labels/val"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__),
                             "csv_files"), exist_ok=True)

    #########################
    ### Generate Annotations
    #########################

    annotation_dir = os.path.join(os.path.dirname(__file__),
                                  "caltechpedestriandataset/annotations/annotations/") + "/*"
    print(annotation_dir)
    classes = ['person']
    number_of_truth_boxes = 0

    img_id_list = []
    label_list = []
    split_list = []
    num_annot_list = []
    
    # Sets
    for sets in tqdm(sorted(glob.glob(annotation_dir))):
        set_id = os.path.basename(sets)
        set_number = int(set_id.replace('set', ''))
        split_dataset = "train" if set_number <=5 else "val"
        
        # Videos
        for vid_annotations in sorted(glob.glob(sets + "/*.vbb")):
            video_id = os.path.splitext(os.path.basename(vid_annotations))[0] # Video ID
            vbb = loadmat(vid_annotations) # Read VBB File
            obj_lists = vbb['A'][0][0][1][0] # Annotation List
            obj_lbl = [str(v[0]) for v in vbb['A'][0][0][4][0]] # Label List
            
            # Frames
            for frame_id, obj in enumerate(obj_lists):
                if(len(obj)>0):
                    # Labels
                    labels = ''
                    num_annot = 0
                    for pedestrian_id, pedestrian_pos in zip(obj['id'][0], obj['pos'][0]):
                        pedestrian_id = int(pedestrian_id[0][0]) - 1 # Pedestrian ID
                        pedestrian_pos = pedestrian_pos[0].tolist() # Pedestrian BBox
                        # class filter and height filter: here example for medium distance
                        if obj_lbl[pedestrian_id] in classes and pedestrian_pos[3] >= 75 and pedestrian_pos[3] <= 250:
                            yolo_box_format = convertBoxFormat(pedestrian_pos) # Convert BBox to YOLO Format
                            labels += '0 ' + ' '.join([str(n) for n in yolo_box_format]) + '\n'
                            num_annot += 1
                            number_of_truth_boxes += 1
                    
                    # Check Labels
                    if not labels:
                        continue

                    image_id = set_id + '_' + video_id + '_' + f"{frame_id:04d}"
                    img_id_list.append(image_id)
                    label_list.append(labels)
                    split_list.append(split_dataset)
                    num_annot_list.append(num_annot)
        
    print("Number of Ground Truth Annotation Box:", number_of_truth_boxes)

    df_caltech_annot = pd.DataFrame({
        "image_id": img_id_list,
        "label": label_list,
        "split": split_list,
        "num_annot": num_annot_list
    })

    df_caltech_annot["set_id"] = df_caltech_annot["image_id"].apply(lambda x: x.split("_")[0])
    df_caltech_annot["video_id"] = df_caltech_annot["image_id"].apply(lambda x: x.split("_")[1])
    df_caltech_annot["frame_id"] = df_caltech_annot["image_id"].apply(lambda x: int(x.split("_")[2]))

    df_caltech_annot.to_csv("csv_files/frame_metadata.csv", index=False)
    # df_caltech_annot

    ######################
    # Filter Image Files
    ######################

    df_set_video = df_caltech_annot.groupby(["set_id", "video_id", "split"])["image_id"].count().reset_index()
    df_set_video = df_set_video.rename(columns={"image_id": "total_image"})

    df_set_video_train = df_set_video[df_set_video["split"]=="train"].reset_index(drop=True)
    df_set_video_val = df_set_video[df_set_video["split"]=="val"].reset_index(drop=True)

    # print(df_set_video_train.head())
    # print(df_set_video_val.head())

    total_train_image = sum(df_set_video_train["total_image"])
    total_val_image = sum(df_set_video_val["total_image"])
    print("Number of Train:", total_train_image)
    print("Number of Val:", total_val_image)

    df_set_video_train = df_set_video_train.groupby("set_id")["video_id"].count().reset_index()
    df_set_video_val = df_set_video_val.groupby("set_id")["video_id"].count().reset_index()
    try:
        df_set_video_count = df_set_video_train.append(df_set_video_val).reset_index(drop=True)
    except:
        df_set_video_count = pd.concat([df_set_video_train, df_set_video_val], ignore_index=True)
    df_set_video_count = df_set_video_count.rename(columns={"video_id": "total_video"})
    # display(df_set_video_count)

    df_train_filtered = pd.DataFrame()
    df_val_filtered = pd.DataFrame()
    set_id_list = list(df_caltech_annot["set_id"].unique())
    for i, set_id in enumerate(set_id_list):
        df_set_id = df_set_video[df_set_video["set_id"]==set_id].reset_index(drop=True)
        video_id_list = list(df_set_id["video_id"].unique())
        
        for j, vid_id in enumerate(video_id_list):
            df_video_id = df_caltech_annot[(df_caltech_annot["set_id"]==set_id)&(df_caltech_annot["video_id"]==vid_id)].reset_index(drop=True)
            frame_total = df_video_id.shape[0]
            if(i<=5): # 10000 Train Images
                limit = int(round((frame_total / total_train_image) * 10000, 0))
                df_video_id = df_video_id[:limit]
                try:
                    df_train_filtered = df_train_filtered.append(df_video_id)
                except:
                    df_train_filtered = pd.concat([df_train_filtered, df_video_id], ignore_index=True)
            else: # 2500 Val Images
                limit = int(round((frame_total / total_val_image) * 2500, 0))
                df_video_id = df_video_id[:limit]
                try:
                    df_val_filtered = df_val_filtered.append(df_video_id)
                except:
                    df_val_filtered = pd.concat([df_val_filtered, df_video_id], ignore_index=True)

    df_train_filtered = df_train_filtered.reset_index(drop=True)
    df_val_filtered = df_val_filtered.reset_index(drop=True)

    # display(df_train_filtered)
    # display(df_val_filtered)

    df_train_filtered.to_csv("csv_files/train_frame_filtered.csv", index=False)
    df_val_filtered.to_csv("csv_files/val_frame_filtered.csv", index=False)

    convert_caltech("Train", df_train_filtered)
    convert_caltech("Test", df_val_filtered)

    generate_labels("train", df_train_filtered)
    generate_labels("val", df_val_filtered)

    # %%writefile /datasets/caltechpedestriandataset.yaml
    # Create Custom Dataset Configuration
    # path: /datasets
    # train: /datasets/images/train
    # val: /datasets/images/val
        
    # nc: 1
        
    # names: [
    #     'person'
    # ]