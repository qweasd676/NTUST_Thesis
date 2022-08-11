
import cv2
import pandas as pd
import numpy as np
import threading
import time
import glob
import os
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from aniposelib.utils import load_pose2d_fnames
from aniposelib.cameras import Camera, CameraGroup



# How to use this code. the code is below.
    # path = './analysis/csv/*.csv'
    # analysis_list = sorted(glob.glob(path))
    # print(analysis_list)
    # deeplabcut_csv_analysis(analysis_list)

#0411 have a bug.

def draw2dpoint_sp():
    
    img = cv2.imread('./2dpredict/0529/image00049.jpg')

    point1 = [866.036489470319,472.554177941988,840.148892239425,461.073000761638,846.82159073569,	434.830651200408,	861.510747588786,	485.216165055009	,820.635839732664	,505.064201281097	,806.46795055696	,522.859265640148,	856.254538536072	,499.804889514273,	804.955603779673,	536.777452580286,	793.673150785023	,581.532162991984	,883.617789007339,	479.745370270973,	910.824272328484,	485.365008880345	,930.295354465141,	498.387631317678,	883.788042902946	,487.83431688481,	917.686187466545	,507.574877925445,	941.591851777362	,505.599633854038	,884.879858107756	,497.435291006392,	904.92700349885	,588.895112292649	,936.554829326893	,625.124216007202	,880.379439000507	,460.876642210602,	837.774307250584	,630.071433755197]
    point2 = [866.194303035736,472.123119354248,840.315425634384,459.580417662859,	845.934492111206	,431.984717130661,	860.581013679504	,486.537690639496	,818.848815202713	,503.751223564148	,805.937264680862	,522.586434841156	,852.756040215492	,506.128280162811	,801.004074573517,	550.188066959381	,797.345376968384	,598.883993148804,	885.079756140709,	480.585288047791	,910.524214506149,	487.279897451401,	927.350236654282,	496.934870481491	,884.313069194555	,488.9242208004	,916.390794217586	,507.663947880268	,941.908849716187	,503.525895357132	,882.218214988709	,499.85746191442	,900.695056736469	,588.880682528019,	936.212407588959,	622.992527723312	,879.399985790253	,461.875247001648	,836.265988260508	,629.664744734764]

    point1_x = point1[::2]
    point1_y = point1[1::2] 


    point2_x = point2[::2]
    point2_y = point2[1::2] 

    img_1 = img.copy()
    img_2 = img.copy()


    for x in range(0,20):
        point_size = 2
        point_color = (0,0,255)
        point_color1 = (0,0,0)
        thickness = 4

        point_GT = [int(point1_x[x]),int(point1_y[x])]
        cv2.circle(img_1,point_GT,point_size,point_color,thickness)


        point_PD = [int(point2_x[x]),int(point2_y[x])]
        cv2.circle(img_2,point_PD,point_size,point_color1,thickness)


    res = cv2.resize(np.hstack((img_1,img_2)),(1600,900))
    cv2.imwrite('aa.jpg',res)

    return 


def draw2dpoint():

    # img_file_orignal = sorted(glob.glob(img_src))
    path_ = './2dpredict/0603/'

    name = ['jump','run','dlc']
    ind = 0

    path = ['{0}CollectedData_{1}_mvs_C.csv'.format(str(path_),name[ind])]

    fname_dict = {
                  'B': '{0}{1}-camB.h5'.format(str(path_),name[ind]),
                  'C': '{0}{1}-camC.h5'.format(str(path_),name[ind])

                }

    print(fname_dict)
    cgroup = CameraGroup.load('{0}calibration_0603_BC.toml'.format(str(path_)))


    d = load_pose2d_fnames(fname_dict, cam_names=cgroup.get_names())
    score_threshold = 0.1
    n_cams, n_points, n_joints, _ = d['points'].shape
    points = d['points']
    scores = d['scores']
    bodyparts = d['bodyparts']
    # remove points that are below threshold
    points[scores < score_threshold] = np.nan
    points_flat = points.reshape(n_cams, -1, 2)
    scores_flat = scores.reshape(n_cams, -1)


    points_flat_copy = points_flat
    pd_csv = pd.read_csv(path[0])
    img_name = pd_csv.iloc[2:,2].values

    GT_2D = pd_csv.iloc[2:,3:].values
    GT_2D_x = GT_2D[:,::2]
    GT_2D_y = GT_2D[:,1::2]
    GT_2D_x = GT_2D_x.flatten()
    GT_2D_y = GT_2D_y.flatten()
    points_flat_copy[0,:,0] = GT_2D_x
    points_flat_copy[0,:,1] = GT_2D_y    

    print(points_flat_copy.shape)
    print(points_flat.shape)


    len_ = [points_flat_copy[1,:,1],points_flat_copy[1,:,0]]

    i = 0
    for x in range(0,points_flat.shape[1],20):

        print('{0}{1}_camC/{2}'.format(str(path_),name[ind],str(img_name[i])))
        
        
        img = cv2.imread('{0}{1}_camC/{2}'.format(str(path_),name[ind],str(img_name[i])))
        i = i +1
        img_GT = img.copy()
        img_PD = img.copy()
        for y in range(0,20):
            point_size = 2
            point_color = (0,0,255)
            point_color1 = (0,0,0)
            thickness = 4
            
            # print(points_flat_copy[0,x+y,0],points_flat[0,x+y,0])

            if np.isnan(points_flat_copy[0,x+y,0]) != True:
                point_GT = [int(points_flat_copy[0,x+y,0]),int(points_flat_copy[0,x+y,1])]
                cv2.circle(img_GT,point_GT,point_size,point_color,thickness)

            if np.isnan(points_flat[0,x+y,0]) != True:
                point_PD = [int(points_flat[0,x+y,0]),int(points_flat[0,x+y,1])]
                cv2.circle(img_PD,point_PD,point_size,point_color1,thickness)


        res = cv2.resize(np.hstack((img_GT,img_PD)),(1600,900))
        cv2.imwrite('{0}predict/jump1/{1}{2}.jpg'.format(path_,name[ind],i),res)
        # cv2.imshow('dsad',res)
        # cv2.waitKey(0)

        # break







    #修改的地方
    # points_flat_copy = points_flat
    # for index,element in enumerate(path):
    #     print(element)
    #     pd_csv = pd.read_csv(element)
    #     GT_2D = pd_csv.iloc[2:,3:].values
    #     GT_2D_x = GT_2D[:,::2]
    #     GT_2D_y = GT_2D[:,1::2]
    #     # GT_2D_x = GT_2D_x.flatten()
    #     # GT_2D_y = GT_2D_y.flatten()
    #     # points_flat_copy[index,:,0] = GT_2D_x
    #     # points_flat_copy[index,:,1] = GT_2D_y

        






    return 


def deeplabcut_csv_analysis(path):
    index_img = 0
    for image_set in path:
        
        pd_csv = pd.read_csv(image_set)
        title_ = pd_csv.iloc[0,1::3].values
        # print(title_)
        likelihood = pd_csv.iloc[2:,3::3].values
        res = np.mean(likelihood, dtype = float, axis = 0)

        size = int(res.shape[0])
        fig,axes=plt.subplots(int(size/5),5 + int(size % 5),figsize = (19,11), dpi = 100)
        # fig.figure(figsize = (8,4))
        # print(int(size/5))
        x = np.arange(1,151,1)
        y_ticks = np.arange(0,1.2,0.1)

        

        for index in range(0,size):
            y = (likelihood[:,index]).astype(float)
            np.set_printoptions(precision=5, suppress=True)
            x_index = int(index%5)
            y_index = int(index/5)

            # print(title_[index])
            axes[y_index,x_index].plot(x,y)
            axes[y_index,x_index].yaxis.set_ticks(y_ticks)
            axes[y_index,x_index].set_xlabel("frame")
            axes[y_index,x_index].set_ylabel("likelihood")
            axes[y_index,x_index].set_title("probability of the {0}".format(str(title_[index])))
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.5)
        plt.savefig('image_analysis_index_{0}.jpg'.format(str(index_img)))
        # plt.close()
        # plt.show()
        index_img += 1
def regression_ml(path):


    X = np.array([
        [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180]
    ])
    y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364])

    lm = LinearRegression()
    lm.fit(X, y)

    # 印出係數
    print(lm.coef_)

    # 印出截距
    print(lm.intercept_ )

    return 0


def analyze_3D_points(path):

    pd_csv = pd.read_csv(path)
    # title = pd_csv.iloc[0,1::3].values
    # print(pd_csv)

    
    # 0   1  2 ->  run  FFL
    # 18 19 20 ->  run  MFL
    # 36 37 38 ->  run  HFL

    # 54 55 56 ->  run  FFR
    # 72 73 74 ->  run  MFR
    # 90 91 92 ->  run  HFR
    number = 38

    for number_ind in range(number):  
        ff_list = [0,18,36,54,72,90]
        name_list = ['Front foot Left','Front foot Right','Middle foot Left','Middle foot Right','Hind foot Left','Hind foot Right']
        index = 0

        for index,element in enumerate(ff_list):
            one = element
            two = one + 1
            three = two + 1

            pos_x1 = (pd_csv.iloc[:number_ind,one].values).astype(float)
            pos_y1 = (pd_csv.iloc[:number_ind,two].values).astype(float)
            pos_z1 = (pd_csv.iloc[:number_ind,three].values).astype(float)

            # print(pos_x1.shape,len(pos_y1),len(pos_z1))
            pos_x2 = (pd_csv.iloc[:number_ind,one+6].values).astype(float)
            pos_y2 = (pd_csv.iloc[:number_ind,two+6].values).astype(float)
            pos_z2 = (pd_csv.iloc[:number_ind,three+6].values).astype(float)

            pos_x3 = (pd_csv.iloc[:number_ind,one+6*2].values).astype(float)
            pos_y3 = (pd_csv.iloc[:number_ind,two+6*2].values).astype(float)
            pos_z3 = (pd_csv.iloc[:number_ind,three+6*2].values).astype(float)

            # print(pos_z1[0],pos_z2[0],pos_z3[0])
                # 建立 3D 圖形
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # 產生 3D 座標資料

            # 繪製 3D 座標點
            ax.scatter(pos_x1, pos_y1, pos_z1, c='r',cmap='Red', marker='^', label='first joint')
            ax.scatter(pos_x2, pos_y2, pos_z2, c='b', cmap='Blue', marker='o', label='Second joint')
            ax.scatter(pos_x3, pos_y3, pos_z3, c='g', cmap='Green', marker='x', label='Third joint')
            ax.plot(pos_x1, pos_y1,pos_z1,color ='r')
            ax.plot(pos_x2, pos_y2,pos_z2,color ='b')
            ax.plot(pos_x3, pos_y3,pos_z3,color ='g')

            # ax.set_title('{0}'.format(str(title[0+3])))
            ax.set_title('{0} run trajectory analysis'.format(str(name_list[index])) )


            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(60,-45)
            # 顯示圖例
            ax.legend()

            # 顯示圖形
            plt.show()
            break


#0425 解決了線性回歸和缺失值的問題
def deal_with_csv(path):
    print(path)
    frames_excel = []
    pd_csv = pd.read_csv(path)
    title = pd_csv.iloc[:2,:]
    # print(title)

    df = pd_csv.iloc[2:,:]
    df = df.astype(float)
    df = df.interpolate(method ='linear', limit_direction ='both', limit = 10)
    # print(df[df.isnull().values==True])

    result = pd.concat([title,df],ignore_index = True)
    # print(result)
    result.to_csv('./csv/result.csv',index = False)

    return 0


def angel_3D_point(path,sel):
    pd_csv = pd.read_csv(path)
    row,col = pd_csv.shape

    name = ['left front foot','left middle foot','right front foot','right middle foot','left hind foot','right hind foot']
    count = 0


    if sel:
        number = 38
        fig,axes=plt.subplots(1,2,figsize = (8,6), dpi = 100)
        angle_1 = (pd_csv.iloc[:number,2].values).astype(float)
        angle_2 = (pd_csv.iloc[:number,5].values).astype(float)

        count = 4
        axes[0].plot(angle_1,'b')
        axes[0].grid(True)
        axes[0].set_title('Angle of the {0}'.format(str(name[count])))
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Angle')

        axes[1].plot(angle_2,'b')
        axes[1].grid(True)
        axes[1].set_title('Angle of the {0}'.format(str(name[count+1])))
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Angle')


    else:
        fig,axes=plt.subplots(3,2,figsize = (12,8), dpi = 100)
        for x in range(0,3):
            angle_1 = (pd_csv.iloc[:,x*2].values).astype(float)
            angle_2 = (pd_csv.iloc[:,x*2+1].values).astype(float)

            axes[x,0].plot(angle_1,'b')
            axes[x,0].grid(True)
            axes[x,0].set_title('angle of {0}'.format(str(name[count])))
            axes[x,0].set_xlabel('frame')
            axes[x,0].set_ylabel('angle')

            count += 1
            axes[x,1].plot(angle_2,'b')
            axes[x,1].grid(True)
            axes[x,1].set_title('angle of {0}'.format(str(name[count])))
            axes[x,1].set_xlabel('frame')
            axes[x,1].set_ylabel('angle')
            # axes[x,1].axhline(y = each_part_size["{0}".format(str(title[y + x*3 + 3]))], c="r", ls="--", lw=2)
            count += 1
    
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.show()

    return


#region
def analyze_3D_feet(path):
    start = 0
    end   = 10
    pd_csv = pd.read_csv(path)
    title = pd_csv.iloc[0,1:].values

    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]

    # print(title)
    each_part_size = {}
    for y in range(0,int(title.shape[0])-6,9):

        # fig,axes=plt.subplots(2,2,figsize = (19,11), dpi = 100)
        fig,axes=plt.subplots(1,2,figsize = (12,6), dpi = 100)

        # print('y:{0}'.format(y))
        # print(title[y],'\n')
        for x in range(0,2):
            
            pos_x = (pd_csv.iloc[2:,(y+1) + x * 3].values).astype(float)
            pos_y = (pd_csv.iloc[2:,(y+2) + x * 3].values).astype(float)
            pos_z = (pd_csv.iloc[2:,(y+3) + x * 3].values).astype(float)

            pos_x1 = (pd_csv.iloc[2:,(y+1) + x * 3 + 3].values).astype(float)
            pos_y1 = (pd_csv.iloc[2:,(y+2) + x * 3 + 3].values).astype(float)
            pos_z1 = (pd_csv.iloc[2:,(y+3) + x * 3 + 3].values).astype(float)

            translation_x = pos_x1 - pos_x
            translation_y = pos_y1 - pos_y
            translation_z = pos_z1 - pos_z
            translation_size = np.sqrt(np.power(translation_x,2) + np.power(translation_y,2) + np.power(translation_z,2) ) 
            # print(translation_x[34],translation_y[34],np.arctan(translation_y[34]/translation_x[34]))
            radians_ = np.arcsin(translation_y/translation_size)
            angle = np.degrees(radians_)
            each_part_size["{0}".format(str(title[y + x*3 + 3]))] = np.nanmean(translation_size)

            # angle = np.degrees(np.arctan(translation_y/translation_x))

            axes[x].plot(angle[20:120],'b')
            axes[x].grid(True)
            # axes[0,x].axvline(x=start, c="r", ls="--", lw=2)
            # axes[0,x].axvline(x=end, c="r", ls="--", lw=2)
            axes[x].set_title('Angle of {0}'.format(str(title[y + x*3 + 3])),fontsize=20)
            axes[x].set_xlabel('Frame',fontsize=15)
            axes[x].set_ylabel('Angle',fontsize=15)

            # axes[1,x].plot(translation_size[20:120],'r')
            # axes[1,x].grid(True)
            # axes[1,x].set_title("length of {0}".format(str(title[y + x*3 + 3])))
            # axes[1,x].axhline(y = each_part_size["{0}".format(str(title[y + x*3 + 3]))], c="r", ls="--", lw=2)
            # axes[1,x].set_xlabel('frame')
            # axes[1,x].set_ylabel('cm')

            # print(len(pos_x))
            # print('\npos_x:',pos_x[start:end])
            # print('\npos_y:',pos_y[start:end])
            # print('\npos_x1:',pos_x1[start:end])
            # print('\npos_y1:',pos_y1[start:end])
            # print('\ntranslation_x:',translation_x[start:end])
            # print('\ntranslation_y:',translation_y[start:end])
            
            # print('\narcsin  :',radians_[start:end])
            # print('\ndegress :',angle[start:end],'\n')
            # print('\ntranslation_size:',translation_size[start:end])
        target = './feet_analysis/'
        plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        plt.savefig( target+ 'image_analysis_angle_{0}.jpg'.format(str(title[y]+ str(filename))))
    print("compete create image to {0}".format(target))
        # plt.show()
    plt.close('all')
        

    # for y in range(int(title.shape[0])-6+1 , int(title.shape[0]),6):
    #     # fig,axes=plt.subplots(1,1,figsize = (19,11), dpi = 100)

    #     head_x = (pd_csv.iloc[2:,(y)].values).astype(float)
    #     head_y = (pd_csv.iloc[2:,(y+1)].values).astype(float)
    #     head_z = (pd_csv.iloc[2:,(y+2)].values).astype(float)

    #     tail_x = (pd_csv.iloc[2:,(y)  +3].values).astype(float)
    #     tail_y = (pd_csv.iloc[2:,(y+1) +3].values).astype(float)
    #     tail_z = (pd_csv.iloc[2:,(y+2) +3].values).astype(float)
    #     diff_x = head_x - tail_x
    #     diff_y = head_y - tail_y
    #     diff_z = head_z - tail_z

    #     head_tail_size = np.sqrt(np.power(diff_x,2) + np.power(diff_y,2) + np.power(diff_z,2) ) 
    #     head_tail_mean = np.nanmean(head_tail_size)
    #     plt.plot(head_tail_size,'b')
    #     plt.grid(True)
    #     plt.xlabel('frame')
    #     plt.ylabel('cm')

    #     plt.axhline( y= head_tail_mean, c="r", ls="--", lw=2)
    #     # print('head_x: ',head_x[start:end],'\n')
    #     # print('head_y: ',head_y[start:end],'\n')
    #     # print('head_z: ',head_z[start:end],'\n')
        
    #     # print('tail_x: ',tail_x[start:end],'\n')
    #     # print('tail_y: ',tail_y[start:end],'\n')
    #     # print('tail_z: ',tail_z[start:end],'\n')
    #     plt.savefig('head_to_tail.jpg')

    #     # plt.show()

#endregion      



def find_error_3d(path):
    # 0   1  2 ->  run  FFL
    # 18 19 20 ->  run  MFL
    # 36 37 38 ->  run  HFL

    # 54 55 56 ->  run  FFR
    # 72 73 74 ->  run  MFR
    # 90 91 92 ->  run  HFR
    pd_csv = pd.read_csv(path)
    number = 35
    for index in range(0,6):
        error_1 = np.nanmean(pd_csv.iloc[:number,index*3*6 +3].values).astype(float)
        error_2 = np.nanmean(pd_csv.iloc[:number,index*3*6 +3+6].values).astype(float)
        error_3 = np.nanmean(pd_csv.iloc[:number,index*3*6 +3+6*2].values).astype(float)
        totol_error = (error_1+error_2+error_3)/3
        print(totol_error)
 
        




