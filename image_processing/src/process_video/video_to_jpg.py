import os
import cv2
import glob


def video_to_pic(video_path,output_folder,type_pic):
    if os.path.isdir(output_folder):
        print("Delete old result folder: {}".format(output_folder))
        os.system("rm -rf {}".format(output_folder))
    os.system("mkdir {}".format(output_folder))
    print("create folder: {}".format(output_folder))
    # print('xx')
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    video = []

    for idx in range(0,frame_count,1):
        vc.set(1, idx)
        ret, frame = vc.read()
        height, width, layers = frame.shape
        size = (width, height)

        if frame is not None:
            file_name = '{}{:08d}.{}'.format(output_folder,idx,type_pic)
            cv2.imwrite(file_name, frame)

        print("\rprocess: {}/{}".format(idx+1 , frame_count), end = '')
    vc.release()

video_path = '/home/ntustee_chen/桌面/image_prcoessing/video/pha.mp4'
output_folder = '/home/ntustee_chen/桌面/image_prcoessing/image/pha/'
type_pic = 'png'
video_to_pic(video_path,output_folder,type_pic)