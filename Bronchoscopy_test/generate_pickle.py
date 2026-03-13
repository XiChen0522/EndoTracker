import cv2
import numpy as np


# pickle file:
# data = pickle.load(f)
# keys: v1, v2, v3
# vid_data = data['v1']
# vid_data['video'] = frames, height, width, 3
# vid_data['occluded'] = points, frames
# vid_data['points'] = points, frames, 2

import torch
import cv2
import numpy as np
import os
import pickle
import glob



def read_images_and_numpy(dir_path):
    """
    Read all images and a NumPy file from a directory and save them as a pickle file.

    Args:
        dir_path (str): Path to the directory containing images and NumPy files.
    """
    images = []
    points = None

    
    
    # Iterate over all files in the directory
    for file_name in sorted(os.listdir(os.path.join(dir_path, 'video'))):
        file_path = os.path.join(dir_path, 'video', file_name)
        # If the file is an image, read it
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)
                
    def read_pickle_file(pickle_file):
        # 读取pickle文件并将其反序列化为Python对象
        with open(pickle_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        return data
    
    # load npy
    # breakpoint()
    points = read_pickle_file(os.path.join(dir_path, 'video', 'points.pkl'))
    points = np.array(points, dtype=np.float32)
    points[:,:,0] = points[:,:,0]/images[0].shape[1]
    points[:,:,1] = points[:,:,1]/images[0].shape[0]
    
    
    occluded = np.full((points.shape[0], points.shape[1]), False, dtype=bool)
    
    
    occluded_pkl = read_pickle_file(os.path.join(dir_path, 'video', 'visibility.pkl'))
    occluded_pkl = np.array(occluded_pkl)
    # print("occluded_pkl",occluded_pkl)
    
    # occluded_pkl=np.array([1, 0, 1])
    # 将 occluded_pkl 转换为 (3, 1)
    occluded_pkl_reshaped = occluded_pkl.reshape(-1, 1)

    # 创建一个与 occluded_pkl 长度一致的数组 [1, 1, 1]
    ones_array = np.ones((occluded_pkl.shape[0], 1), dtype=occluded_pkl.dtype)

    # 将原数组与新的列拼接
    occluded_pkl = np.hstack((ones_array,occluded_pkl_reshaped ))
    # print("occluded_pkl",occluded_pkl)
    # 找到 occluded_pkl 中值为 1 的索引
    indices = np.where(occluded_pkl[:, 1] == 0)[0]

    # 在 occluded 的对应位置设为 True
    occluded[indices, 1] = True
    # print(occluded)
    # print("occluded",occluded)
    return images, occluded, points

    
if __name__ == "__main__":
    
    # data = {}
    #  Open the file in read mode
    with open('vid_dir_list_full.txt', 'r') as file:
        # Iterate over each line in the file
        for line in file:
            line = line.replace('/Bronchoscopy_test/', '')
            vid_data = {}
            dir_path = line.strip()
            # print(dir_path)

            # Read images and numpy data and save them as a pickle file
            frames, vis, pts = read_images_and_numpy(dir_path)
            
            # Save the frames and numpy data to the dictionary using folder name as the key
            # data[(dir_path.split('/')[-2]+'_'+dir_path.split('/')[-1])] = {
            #     'video': np.array(frames),
            #     'occluded': vis,
            #     'points': pts
            # }
            # break

            vid_data['video'] = np.array(frames)
            vid_data['occluded'] = vis
            vid_data['points'] = pts
        
            # Save the complete data dictionary to a pickle file
            with open(dir_path+'/pt_gt.pkl', 'wb') as pickle_file:
                pickle.dump(vid_data, pickle_file)
                
            # break