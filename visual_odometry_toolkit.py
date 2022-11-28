import tkinter as tk
from tkinter.ttk import Frame, Label, Entry, Button
from PIL import Image, ImageTk
import time
from glob import glob
import cv2
import numpy as np
import os
from pose_evaluation_utils import *

def return_calibration_intrinsics(calib_path):
    myvars = {}
    with open(calib_path) as myfile:
        for line in myfile:
            name, var = line.partition(":")[::2]
            var_list = [float(x) for x in var.split()]
            myvars[name.strip()] = var_list
    lst = myvars["P0"]
    fx = lst[0]
    cx = lst[2]
    fy = lst[5]
    cy = lst[6]
    return fx,fy,cx,cy

def return_gt_dataframe(gt_path):
    file = open(gt_path, 'r')
    lines = file.readlines()
    lines_lst = []
    for l in lines:
        tlst = l.split()
        tlst = [float(t) for t in tlst]
        lines_lst.append(tlst)
    return lines_lst

def changeValue():
    label1 = tk.Label(window, text="image")
    label2 = tk.Label(window,text="feature image")
    label3 = tk.Label(window, text="feature match")
    label4 = tk.Label(window, text="trajectory map")

    text_label1 = tk.Label(window, text="Image sequence").place(x=20, y=20)
    text_label2 = tk.Label(window, text="Features detected").place(x=20, y=354)
    text_label1 = tk.Label(window, text="Feature matching").place(x=20, y=688)
    text_label1 = tk.Label(window, text="Trajectory map").place(x=980, y=20)

    # print("val = ", clicked.get())
    # input_label3.place_forget()

    folder_path = "Dataset/KITTI/sequences/" + kitti_seq_num.get() + "/image_0"
    gt_path = "Dataset/KITTI/poses/" + kitti_seq_num.get() + ".txt"
    calib_path = "Dataset/KITTI/sequences/" + kitti_seq_num.get() + "/calib.txt"

    len_trajectory_map = int(len_traj_map.get())

    width, height = 1241.0, 376.0
    fx,fy,cx,cy = return_calibration_intrinsics(calib_path)
    matrix = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape((3,3))

    trajectory_map = np.zeros((len_trajectory_map, len_trajectory_map, 3), dtype = np.uint8)
    pose_file = "./KITTI-" + kitti_seq_num.get() + "-pose_file.txt" 

    if kitti_seq_num.get() in gt_available:
        is_gt_available = True
    else:
        is_gt_available = False

    if plot_gt.get() == "YES" and is_gt_available:
        gt_df = return_gt_dataframe(gt_path)

    orb = cv2.ORB_create(nfeatures=6000)
    brisk = cv2.BRISK_create(thresh = 20)

    if feature_detector.get() == "ORB":
        fd = orb
    elif feature_detector.get() == "BRISK":
        fd = brisk
    else:
        print("Invalid feature descriptor")

    img_list = glob(folder_path + '/*.png')
    img_list.sort()
    num_frames = len(img_list)
    # print(img_list)
    print(num_frames)

    for i in range(num_frames):
        current_img = cv2.imread(img_list[i], 0)

        if i == 0:
            current_R = np.eye(3)
            current_T = np.array([[0],[0],[0]])
        else:
            previous_img = cv2.imread(img_list[i-1], 0)

            keypoint1, descriptor1 = fd.detectAndCompute(previous_img, None)
            keypoint2, descriptor2 = fd.detectAndCompute(current_img, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(descriptor1, descriptor2)

            matches = sorted(matches, key = lambda x: x.distance)

            initial_feature_match = cv2.drawMatches(previous_img, keypoint1, current_img, keypoint2, matches[0:100], None)

            points1 = np.float32([keypoint1[m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoint2[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=matrix, method=cv2.RANSAC, prob=0.999, threshold=1)
            points1 = points1[mask.ravel() == 1]
            points2 = points2[mask.ravel() == 1]
            _, R, T, mask = cv2.recoverPose(E, points1, points2, cameraMatrix=matrix)

            R = R.transpose()
            T = -np.matmul(R, T)

            if i == 1:
                current_R = R
                current_T = T
            else:
                current_R = np.matmul(previous_R, R)
                current_T = np.matmul(previous_R, T) + previous_T

            keypoint_img = cv2.drawKeypoints(current_img, keypoint2, None, color=(0,255,0), flags = 0)

            # Update trajectory map
            offset = (int(len_trajectory_map/2))
            
            if plot_gt.get() == "YES" and is_gt_available:
                cv2.circle(trajectory_map, (int(gt_df[i][3])+offset, int(gt_df[i][11])+offset), 1, (0,255,0), 2)

            cv2.circle(trajectory_map, (int(current_T[0])+offset, int(current_T[2])+offset), 1, (255,0,0), 2)

            present_trajectory_map = cv2.rotate(trajectory_map.copy(), cv2.ROTATE_90_CLOCKWISE)
            present_trajectory_map = cv2.rotate(present_trajectory_map, cv2.ROTATE_90_CLOCKWISE)
            present_trajectory_map = cv2.flip(present_trajectory_map, 1)

            print("iteration", i)
            resized_down = cv2.resize(current_img.copy(), (940,284), interpolation= cv2.INTER_AREA)
            im = Image.fromarray(resized_down)
            img1 = ImageTk.PhotoImage(image=im)
            label1.config(image=img1)
            label1.image = img1
            label1.place(x=20, y=50)

            resized_down = cv2.resize(keypoint_img.copy(), (940,284), interpolation= cv2.INTER_AREA)
            im = Image.fromarray(resized_down)
            img1 = ImageTk.PhotoImage(image=im)
            label2.config(image=img1)
            label2.image = img1
            label2.place(x=20, y=384)
            
            resized_down = cv2.resize(initial_feature_match.copy(), (1880,290), interpolation= cv2.INTER_AREA)
            im = Image.fromarray(resized_down)
            img1 = ImageTk.PhotoImage(image=im)
            label3.config(image=img1)
            label3.image = img1
            label3.place(x=20, y=718)

            resized_down = cv2.resize(present_trajectory_map.copy(), (618,618), interpolation= cv2.INTER_AREA)
            im = Image.fromarray(resized_down)
            img1 = ImageTk.PhotoImage(image=im)
            label4.config(image=img1)
            label4.image = img1
            label4.place(x=980, y=50)

            window.update()
            time.sleep(0.25)
        
        # Save the current pose
        [tx, ty, tz] = [current_T[0], current_T[1], current_T[2]]
        qw, qx, qy, qz = rot2quat(current_R)

        with open(pose_file,'a') as f:
            f.write("%f %f %f %f %f %f %f %f\n" % (0.0, tx, ty, tz, qw, qx, qy, qz))

        previous_R = current_R
        previous_T = current_T

    present_trajectory_map = cv2.rotate(trajectory_map.copy(), cv2.ROTATE_90_CLOCKWISE)
    present_trajectory_map = cv2.rotate(present_trajectory_map, cv2.ROTATE_90_CLOCKWISE)
    present_trajectory_map = cv2.flip(present_trajectory_map, 1)
    present_trajectory_map = cv2.cvtColor(present_trajectory_map, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./KITTI-" + kitti_seq_num.get() + "-trajectory_map.png", present_trajectory_map)

window = tk.Tk()
window.title("Visual Odometry Toolkit for KITTI Dataset")

window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
print("window dimensions=",(window.winfo_screenwidth(), window.winfo_screenheight()))
window.resizable(False,False)

gt_available = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

title_label = tk.Label(window, text="INPUT")
title_label.place(x = 1618, y = 20)
title_label.config(font=("Helvetica", 18))

input_label1 = tk.Label(window, text="Select KITTI Sequence number")
input_label1.place(x = 1618, y = 60)
kitti_seq_num = tk.StringVar()
kitti_seq_num.set("00")
kitti_seq_num_options = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", 
"12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
drop = tk.OptionMenu(window, kitti_seq_num, *kitti_seq_num_options)
drop.place(x=1618, y=90)

input_label2 = tk.Label(window, text="Select feature detector")
input_label2.place(x = 1618, y = 140)
feature_detector = tk.StringVar()
feature_detector.set("ORB")
feature_detector_options = ["ORB", "BRISK"]
drop2 = tk.OptionMenu(window, feature_detector, *feature_detector_options)
drop2.place(x=1618, y=170)

input_label3 = tk.Label(window, text="Select length of trajectory map")
input_label3.place(x = 1618, y = 220)
len_traj_map = tk.StringVar()
len_traj_map.set("1000")
len_traj_map_options = ["250", "500", "750", "1000", "1500", "2000"]
drop3 = tk.OptionMenu(window, len_traj_map, *len_traj_map_options)
drop3.place(x=1618, y=250)

input_label4 = tk.Label(window, text="Plot ground truth")
input_label4.place(x = 1618, y = 300)
plot_gt = tk.StringVar()
plot_gt.set("NO")
plot_gt_options = ["YES", "NO"]
drop4 = tk.OptionMenu(window, plot_gt, *plot_gt_options)
drop4.place(x=1618, y=330)

color_code_label = tk.Label(window)
color_code_img = cv2.imread("icons/color-code.png")
color_code_img = cv2.cvtColor(color_code_img, cv2.COLOR_BGR2RGB)
resized_down = cv2.resize(color_code_img.copy(), (274,51), interpolation= cv2.INTER_AREA)
im = Image.fromarray(resized_down)
img1 = ImageTk.PhotoImage(image=im)
color_code_label.config(image=img1)
color_code_label.image = img1
color_code_label.place(x=1618, y=380)

submit_icon = tk.PhotoImage(file="icons/submit-button.png") 
btn = Button(window, text="Submit", command=changeValue)
btn.config(image=submit_icon)
btn.place(x=1618, y=450)

window.mainloop()