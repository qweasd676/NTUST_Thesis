import os
import glob


# rename pictures, the simple code is below.
    # path = '/home/ntustee_chen/Desktop/DeepLabCut/training_data/deeplabcut_0310/cam/cam2'
    # cameratype = 'cam2'
    # rename_function(path,cameratype)
    # path = '/home/ntustee_chen/Desktop/DeepLabCut/training_data/deeplabcut_0310/cam/cam1'
    # cameratype = 'cam1'
    # rename_function(path,cameratype)


def rename_function(path_name,cameratype,type_image):

    i = 0
    # video_path = sorted(glob.glob(path_name + cameratype + '/*.'+type_image)) 
    video_path = sorted(glob.glob(path_name+ '/*.'+type_image)) 


    # path_name = str(path_name + 'new/'+ cameratype) 
    path_name = str('./locust/new')


    print("The total is {0}".format(len(video_path)))
    print("save location:{0}".format(path_name))




    for item in range(0,len(video_path)):
        # print(video_path[item])
        # os.rename(item,(path_name+'/' + '{0:08d}.png'.format(i)) )
        
        os.rename(video_path[item],(path_name+'/img{0:03d}.png'.format(i)) )

        i += 1
    print("Final the total is {0}".format(i))