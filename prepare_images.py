'''
Preparation of CMU Panoptic derived training data for WHENet

Two steps, one for mtc_dataset (CMU Dome 'Range of Motion' dataset) and on
for the CMU Dome 'Haggling' dataset.

Download the data for the Monocular Total Capture (MTC) from:

http://domedb.perception.cs.cmu.edu/mtc.html
http://posefs1.perception.cs.cmu.edu/mtc/mtc_dataset.tar.gz

It's big. 270GB. 

Download the data for the Haggling data using the PanopticToolbox
https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox

Use the getData.sh script as described at the above link to retrieve
the sequences corresponding to the HD cameras (indices 0-31)

````
getData.sh <sequence_name> 0 31
````

Get all the Haggling data. It will be placed within the directory
that the panoptic toolbox was downloaded to in subdirectories 
named according to the sequence name.

For each sequence, extract the hdFace3d.tar files in place which 
will create a hdFace3d directory.

In the entry point at the bottom of the file, set the root directory
for the panoptic data, either mtc or haggling, containing the
sequences.  Set the output path and, if you are regenerating data,
remove the annotation.txt file (otherwise it will be appended to.)

Choose whether you want to process mtc data or haggling data with 
the 'do_mtc' flag, setting it to True for mtc and false for haggling.

Final data cleanup and deskewing
================================

300W-LP has approximately 150k images but has fewer images with 
yaws around 0 than other angles

MTC range of motion data has around 100k images, and only one
subject per video so that there are no occlusions

Haggling data has more data, but these contain multiple subjects
who may occlude each other.

To generate a final dataset, the range of motion images are 
augmented with haggling images where both have yaw-ranges 
outside -99-99 until the total is the same as 300W-LP. Then 
for small yaws, images are sampled from haggling and range of motion
until the histogram of yaws is level. Throughout this process
images with pitch/roll outside -90-90 are excluded.

The annotation for the included examples are prepended with 
either PANOPTIC or 300W to indicate which dataset they came
from and the annotation file is then split into train and 
test splits which are saved as combine_train and combine_valid
and stored in the panoptic dataset root.
'''
import cv2
import os
import json
import numpy as np
from utils import projectPoints, align, rotationMatrixToEulerAngles2, reference_head, get_sphere, select_euler, inverse_rotate_zyx
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import subprocess

model_points, _ = reference_head(scale=1, pyr=(0., 0., 0.))
kp_idx = np.asarray([17, 21, 26, 22, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8])
kp_idx_model = np.asarray([38, 34, 33, 29, 13, 17, 25, 21, 54, 50, 43, 39, 45, 6])
sphere = []
for theta in range(0, 360, 10):
    for phi in range(0, 180, 10):
        sphere.append(get_sphere(theta, phi, 22))
sphere = np.asarray(sphere)
sphere = sphere + [0, 5, -5]
sphere = sphere.T


def last_8chars(x):
    x = x[-12:]
    x = x.split(".")[0]
    # print(x)
    return (x)


without_top = [0, 3, 5, 8, 9, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 29]
without_top = list(range(31))


def save_img_head(frame, save_path, seq, cam, cam_id, json_file, frame_id, threshold, yaw_ref):
    img_path = os.path.join(save_path, seq)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    # print(frame.size)
    E_ref = np.mat([[1, 0, 0, 0.],
                    [0, -1, 0, 0],
                    [0, 0, -1, 50],
                    [0, 0, 0,  1]])
    cam['K'] = np.mat(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.mat(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3, 1))
    with open(json_file) as dfile:
        fframe = json.load(dfile)
        count_face = -1
        yaw_avg = 0
    for face in fframe['people']:
        # 3D Face has 70 3D joints, stored as an array [x1,y1,z1,x2,y2,z2,...]
        face3d = np.array(face['face70']['landmarks']).reshape((-1, 3)).transpose()
        face_conf = np.asarray(face['face70']['averageScore'])
        landmarks_visibility_count = sum([cam_id in visibility for visibility in face['face70']['visibility']])
        model_points_3D = np.ones((4, 58), dtype=np.float32)
        model_points_3D[0:3] = model_points
        clean_match = (face_conf[kp_idx] > 0.1)  # only pick points confidence higher than 0.1
        kp_idx_clean = kp_idx[clean_match]
        kp_idx_model_clean = kp_idx_model[clean_match]
        if (len(kp_idx_clean) > 6):
            count_face += 1
            rotation, translation, error, scale = align(np.mat(model_points_3D[0:3, kp_idx_model_clean]),
                                                        np.mat(face3d[:, kp_idx_clean]))
            sphere_new = scale * rotation @ (sphere) + translation
            pt_helmet = projectPoints(sphere_new,
                                      cam['K'], cam['R'], cam['t'],
                                      cam['distCoef'])
            temp = np.zeros((4, 4))
            temp[0:3, 0:3] = rotation
            temp[0:3, 3:4] = translation
            temp[3, 3] = 1
            E_virt = np.linalg.inv(temp @ np.linalg.inv(E_ref))
            E_real = np.zeros((4, 4))
            E_real[0:3, 0:3] = cam['R']
            E_real[0:3, 3:4] = cam['t']
            E_real[3, 3] = 1

            compound = E_real @ np.linalg.inv(E_virt)
            status, [pitch, yaw, roll] = select_euler(np.rad2deg(inverse_rotate_zyx(compound)))
            yaw = -yaw
            roll = -roll
            yaw_avg = yaw_avg+yaw
            if (abs(yaw-yaw_ref) > threshold or yaw_ref == -999):
                if (status == True):
                    x_min = int(max(min(pt_helmet[0, :]), 0))
                    y_min = int(max(min(pt_helmet[1, :]), 0))
                    x_max = int(min(max(pt_helmet[0, :]), frame.size[0]))
                    y_max = int(min(max(pt_helmet[1, :]), frame.size[1]))
                    # print(x_min, y_min, x_max, y_max)
                    if (x_min < x_max and y_min < y_max and abs(x_min-x_max) < frame.size[0]):  # some sanity check
                        h = y_max-y_min
                        w = x_max-x_min
                        if not (h/w > 2 or w/h > 2):  # eleminate those too wide or too narrow
                            img = frame.crop((x_min, y_min, x_max, y_max))
                            # draw = ImageDraw.Draw(img)
                            # draw.text((0, 10), "yaw: {}".format(round(yaw)), (0, 255, 255))
                            # draw.text((0, 0), "pitch: {}".format(round(pitch)), (0, 255, 255))
                            # draw.text((0, 20), "roll: {}".format(round(roll)), (0, 255, 255))
                            # plt.imshow(img)
                            # plt.show()
                            filename = '{0:02d}_{1:01d}_{2:02d}_{3:08d}.jpg'.format(cam_id, count_face, landmarks_visibility_count, frame_id)
                            annoname = '{0:02d}_{1:01d}_{2:02d}_{3:08d}.json'.format(cam_id, count_face, landmarks_visibility_count, frame_id)
                            if not (os.path.exists(img_path)):
                                os.mkdir(img_path)
                            file_path = os.path.join(img_path, filename)
                            anno_path = os.path.join(img_path, annoname)
                            img.save(file_path, "JPEG")
                            with open(anno_path, 'w') as f:
                                f.write(json.dumps({'roll': roll, 'yaw': yaw, 'pitch': pitch, 'vis_count': landmarks_visibility_count}))
    if count_face != -1:
        return yaw_avg/(count_face+1)
    else:
        return -999


def sample_video(root_path, sequence_name, save_path, thresh=5, interval=10):
    video_path = os.path.join(root_path, sequence_name, 'hdVideos')
    json_path = os.path.join(root_path, sequence_name, 'hdFace3d')
    # save_path = os.path.join(save_path, sequence_name)
    file_list = os.listdir(json_path)
    json_list = []

    with open(root_path+'/'+sequence_name+'/calibration_{0}.json'.format(sequence_name)) as cfile:
        calib = json.load(cfile)
    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}
    for filename in sorted(file_list, key=last_8chars):
        if filename.endswith('json'):
            json_list.append(os.path.join(json_path, filename))
    start_frame = int(json_list[0][-12:].split(".")[0])
    end_frame = int(json_list[-1][-12:].split(".")[0])
    print(f'processing from frame {start_frame} to {end_frame} step {interval}')

    for i in tqdm(without_top, postfix=sequence_name):  # 0 to 30 hd cameras
        clip = 'hd_00_{0:02d}.mp4'.format(i)
        video_clip = os.path.join(video_path, clip)
        cap = cv2.VideoCapture(video_clip)
        total_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        to_extract_interval = list(range(0, int(total_frame_count), interval))
        count = 0
        json_list_idx = 0
        yaw_prev = -999  # initial value
        with tqdm(total=len(json_list), leave=False) as p_bar:
            while json_list_idx < len(json_list):
                ret, frame = cap.read()
                if not ret:
                    break
                json_file = json_list[json_list_idx]
                next_frame_num = int(json_file[-12:].split(".")[0])
                count += 1
                if (count == next_frame_num):
                    # print('i',i)
                    # print('json',frame_id-start_frame)
                    # print('{}: {}'.format(sequence_name,frame_id))
                    try:
                        if (next_frame_num in to_extract_interval) and not (frame == frame[0, 0]).all():
                            yaw_prev = save_img_head(frame, save_path, sequence_name, cameras[(0, i)], i, json_file, next_frame_num, thresh, yaw_prev)
                    except:
                        break
                    json_list_idx += 1
                    p_bar.update(1)


def mtc_dataset(root_path, sequence_name, save_path, thresh=5):
    img_path = os.path.join(root_path, 'hdImgs', sequence_name)
    json_path = os.path.join(root_path, 'config', sequence_name, 'hdFace3d')
    img_dir_list = os.listdir(img_path)
    img_dir_list = sorted(img_dir_list, key=last_8chars)

    file_list = os.listdir(json_path)
    json_list = []
    for filename in sorted(file_list, key=last_8chars):
        json_list.append(os.path.join(json_path, filename))
    start_frame = int(json_list[0][-12:].split(".")[0])
    # temp = json_list[-1][-12:]
    end_frame = int(json_list[-1][-12:].split(".")[0])

    with open(root_path+'/config/'+sequence_name+'/calibration_{0}.json'.format(sequence_name)) as cfile:
        calib = json.load(cfile)
    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    for cam_n in range(0, 31):
        yaw_prev = -999
        for i in range(len(img_dir_list)):
            curr_dir = os.path.join(img_path, img_dir_list[i])
            # print(os.listdir(curr_dir))
            frame_id = int(img_dir_list[i])
            frame_file_name = "00_{0:02d}_{1:08d}.jpg".format(cam_n, frame_id)
            # print(frame_file_name)
            frame_file_path = os.path.join(curr_dir, frame_file_name)
            if (os.path.isfile(frame_file_path)):
                print(frame_file_name, ' exists!!')
                frame = cv2.imread(os.path.join(frame_file_path))
                yaw_prev = save_img_head(frame, save_path, sequence_name, cameras[(0, cam_n)], cam_n, json_list[frame_id - start_frame], frame_id, thresh, yaw_prev)


def run_download_script(sequence_name):
    print(f'Start download {sequence_name}')
    subprocess.run(['/app/panoptic-toolbox/scripts/getData.sh', sequence_name])
    subprocess.run(['tar', '-xf', f'{sequence_name}/hdFace3d.tar', '-C', sequence_name])
    print(f'{sequence_name} download done')


if __name__ == '__main__':
    root = '/app/panoptic-toolbox'
    root = '/app/HeadPoseEstimation-WHENet'
    out_path = '/app/panoptic-toolbox'
    out_path = '/datasets/Panoptic'
    try:
        os.makedirs(out_path)
    except:
        pass

    do_mtc = False

    if do_mtc:
        seq_list = ['171026_pose1', '171026_pose2', '171026_pose3', '171204_pose1', '171204_pose2', '171204_pose3', '171204_pose4', '171204_pose5', '171204_pose6']
        for i in range(0, 1):
            mtc_dataset(root, seq_list[i], out_path)
    else:
        vid_seq_list = [
            # '171204_pose1',  # done
            # '171204_pose2',  # done
            # '171204_pose3',  # done
            # '171204_pose4',  # done
            # '171204_pose5',  # done
            # '171204_pose6',  # done
            # '171026_pose1',  # done
            # '171026_pose2',  # done
            # '171026_pose3',  # done
            # '170404_haggling_a1',  # done
            # '170404_haggling_a2',  # done
            # '170404_haggling_a3',  # done
            # '170404_haggling_b1',  # done
            # '170404_haggling_b2',  # done
            # '170404_haggling_b3',  # done
            # '170407_haggling_a1',  # done
            # '170407_haggling_a2',  # done
            # '170407_haggling_a3',  # done
            # '170407_haggling_b1',  # done
            # '170407_haggling_b2',  # done
            # '170407_haggling_b3',  # done
            # '170221_haggling_m1',  # done
            # '170221_haggling_m2',  # done
            # '170221_haggling_m3',  # done
            # '170221_haggling_b1',  # done
            # '170221_haggling_b2',  # done
            # '170221_haggling_b3',  # done
            # '170224_haggling_a1',  # done
            # '170224_haggling_a2',  # done
            # '170224_haggling_a3',  # done
            # '170224_haggling_b1',  # done
            # '170224_haggling_b2',  # done
            # '170224_haggling_b3',  # done
            # '170228_haggling_a1',  # done
            # '170228_haggling_a2',  # done
            # '170228_haggling_a3',  # done
            # '170228_haggling_b1',  # done
            # '170228_haggling_b2',  # done
            # '170228_haggling_b3',  # done
            # '160224_haggling1',  # done
            # '160226_haggling1',  # done
            # '160422_haggling1',  # done
            # '161202_haggling1',  # done
            # '160422_ultimatum1',  # done
            # '171026_cello3',  # done
            # '161029_flute1',  # done
            # '161029_piano1',  # done
            # '161029_piano4',  # done
            # '160906_band4',  # done
            # '160906_ian1',  # done
            # '170915_toddler5',  # done
            # '170915_office1',  # done
            # '170407_office2',  # done
            # '161029_build1',  # done
            # '161029_sports1',  # done
        ]
        # vid_seq_list = ['170407_haggling_a1']
        for vid_seq in vid_seq_list:
            try:
                os.path.makedirs(out_path + '/' + vid_seq)
            except:
                pass
            run_download_script(vid_seq)
            sample_video(root, vid_seq, out_path, interval=10)
            shutil.rmtree(os.path.join(root, vid_seq), ignore_errors=True)
