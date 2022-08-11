# NTUST_Thesis
# 基於深度學習之3D蚱蜢運動研究
# Study of 3D grasshopper motion with deep learning

本論文的成果網站：https://sites.google.com/view/asrlabinsectdataset
有任何想詢問的歡迎寄信詢問：a0939686663@gmail.com

All program codes：

1. C++環境： 
    1. Tiscamera_cpp
2. Arduino環境：
    1. Arduino
3. Python環境：
    1. BackgroundMattingV2
    2. Deeplabcut
    3. image_processing

安裝環境須知：

1. C++部分：
    1. 安裝opencv與g++ gcc 即可執行。
2. Arduino部分：
    1. 安裝好Arduino IDE即可執行程式。
3. Python部分：
    1. BackgroundMattingV2：
        1. kornia==0.4.1
        2. tensorboard==2.3.0
        3. torch==1.7.0
        4. torchvision==0.8.1
        5. tqdm==4.51.0
        6. opencv-python==4.4.0.44
        7. onnxruntime==1.6.0
    2. Deeplabcut：
        1. [資料夾內部有 setup.py](http://資料夾內部有setup.py) 安裝到指定版本。
    3. image_processing：
        1. 最重要是安裝opencv 4.0版本以上即可。

碩論流程操作：

1. 建立高速相機環境: 
    1. Create an environment: [https://www.theimagingsource.com/documentation/tiscamera/tutorial.html#setup](https://www.theimagingsource.com/documentation/tiscamera/tutorial.html#setup)
    2. [High-speed camera(dfk 33ux273)](https://www.notion.so/High-speed-camera-dfk-33ux273-0dee94b951084fbfba7ccec3168c11e0) → tutorial
2. 使用高速相機SDK環境進行拍攝:
    1. 首先先使用tcam-capture開啟相機，調整相機參數。
        1. 可使用./tiscamera_cpp/run.sh腳本開啟。
    2. 使用高速相機拍攝蚱蜢資料集，首先先cd ./trigger_save_image。
        1. 首先你必須會使用c++，詳細作請看README.md。
        2. 使用Arduino Nano程式，在裡面可修改脈波頻率。
        3. 開啟./build/execute.sh，裡面可以修改要使用幾台高速相機。
            1. 目前一共有三台，包含47124104, 47124105, 47124106。
        4. 因為我們拍攝都是raw data，必須透過Bayer filter轉換成RGB圖片。
            1. 使用./build/ Bayer2RGB.cpp，指定目標檔案進行轉換成RGB圖。
    3. 使用高速相機拍攝相機校正所需要的圖片，cd ./calibration_save_image 。
        1. 開啟./build/execute.sh，將標定版放入相機範圍內，按下enter進行拍攝。
3. 資料前處理：
    1. 首先安裝 https://github.com/PeterL1n/BackgroundMattingV2 的環境，我們要把拍攝到的照片前景背景分離。
        1. 使用run.sh，裡面可以修改模型參數，修改你的來源影片以及要輸出的結果，由於我們是使用需要背景圖的AI模型，所以需要拍攝一張背景圖。
    2. 接著我們要將前後景分離完的影片進行影像處理，首先cd ./image_processing。
        1. 在main.pay裡，呼叫了opencv.py，其主要功能是將蚱蜢的輪廓進行優化，如修復遺失的邊緣輪廓以及過濾雜訊。
        2. 而在./src中，有image_to_video.py, video_to_jpg.py以及rename_image.py處理資料格式。
4.   2D動物姿態評估DLC(DeepLabCut)，請先安裝 https://github.com/DeepLabCut/DeepLabCut 的環境:
    1. 請先把【2.5小时入门DeepLabCut (1~6) Overview-哔哩哔哩】 [https://b23.tv/TdCh5Uo](https://b23.tv/TdCh5Uo) 系列的影片看完對DLC有基礎的了解。
    2. Cd ./jump_b0/locust-jump_mvs-2022-06-19．
        1. 在DLC中有分GUI和juypterlab去進行。
        2. 首先我們先定義好config.yaml，定義蚱蜢的身體部分, 模型的參數設定，請依照本檔案範例config.yaml或是上面教學影片操作一遍，等熟悉之後在自定義自己的參數。
        3. 使用run_GUI.bash，開啟DLC介面選定自己定義好的config.yaml，我在這邊只是用到介面所提供的label功能進行標注資料集。
        4. 標注完後，我們將整個資料夾壓縮傳到SW930伺服器中，進到符合DLC訓練環境的env，開啟deeplabcut.ipynb，將路徑位置改成你當下的位置，並透過程式裡面的步驟逐步執行，最終等待下練完之後，將整個資料夾全部壓縮傳回至原本的電腦。
        5. 如何使用DGX進行訓練：[DGX Manual](https://www.notion.so/DGX-Manual-c8cec39658f14ce0986399af6598753c)。DGX不像是SW930有廠商提供的軟體去管理，全部都是我們管理人員自己寫的，有任何問題一律請找他們，不地擅自使用。
        6. 如何使用SW930，這邊不會提供詳細的建立流程，請找管理伺服器的同學取得帳號與網址以及提供你的需求，與DGX使用的juypterlab是相同的概念。
5. 3D模型分析：
    1. 我們得到2D的特徵點要重建成3D模型，我們使用支持DLC的anipose library，請先在環境安裝anipose相關的套件。
        1. 首先cd ./cam_mvs中，裡面有config.toml，請參考裡面的設定方式，選擇自己訓練完之後的DLC的config.yaml位置，以及下面有相機校正, 定義骨架以及角度分析方式。
        2. 在run.bash中，我們有相機校正, 2D分析, 三角化重建, 影片合成, 角度分析，cd ./trial_1中，請依照video-raw, calibration放入目標影片以及校正的影片，其可執行後續的功能。
    2. 最終得到的3d資訊我們要進行分析：
        1. 再回到cd ./image_processing，開啟main.py呼叫Optimze_result.py。
        2. 假設3D模型有missing value，我們使用def deal_with_csv()進行修復。
        3. 接著分析3D模型的角度, 運動軌跡。

備註：請先依照我提供config.toml和config.yaml的格式設定，並能後實現出來再去定義自己的資料。
