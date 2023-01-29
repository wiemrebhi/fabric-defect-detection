In this folder there are 7 files :
1-fabric_defect_detection.py : 
Description:
This script captures frames from the video file "video4-fabric2_edited.mp4" and uses background subtraction to detect potential defects in the textile. The script first resizes the frames to 700x700 pixels , then converts the frames to grayscale. It then applies background subtraction to the grayscale frame and creates a binary image by thresholding the resulting foreground mask. It then finds contours in the binary image and draws rectangles around contours that are larger than a certain size. The script then displays each frame with the rectangles drawn around the defects and allows the user to exit by pressing the 'q' key.
Results: 
This methode detect approximately all the stain that are bigger than 100 pixels but it detect also others shapes not in the fabric like the man moving in the background
2-stain_pink_fabric.py :
Description :
This script captures frames from the video file "video4-fabric2_edited.mp4" and uses color thresholding to detect potential stains in the textile. The script first resizes the frames to 700x700 pixels, then converts the frames to HSV color space. The script then defines lower and upper bounds of the color of pink fabric in HSV color space, and creates a binary mask for pink fabric by thresholding the frame. The script also defines lower and upper bounds of the color of black stain in HSV color space, and creates a binary mask for black stain by thresholding the frame. The script then performs morphological operations to remove noise and fill in small gaps, and combine the binary masks for pink fabric and black stains. Then it finds contours in the mask and draws rectangles around contours that are larger than a certain size. The script then displays each frame with the rectangles drawn around the stains and allows the user to exit by pressing the 'q' key.
Results :
this methode of computer vision detect black surface in pink fabric .It gives a good results but it still detect all the pink surfaces 
3-Model.py :
This code is using the VGG19 model, which is a pre-trained convolutional neural network for image classification, and fine-tuning it for a binary classification task of detecting defects in images. The last few layers of the VGG19 model are removed and replaced with new trainable layers, including two dense layers with 2 neurons, which correspond to the two classes (defect or no defect). The model is then trained on a dataset of images from kaggle (source= https://www.kaggle.com/datasets/priemshpathirana/fabric-stain-dataset)add to it some images to make it balance .
4-textile.h5 : contains the trained model 
5-fabric_defect_model_detection.py
Description:
This code loads the pre-trained model called "textile.h5". It then captures video frames from the video file "video4-fabric2_edited.mp4" . For each frame, it applies a sharpen filter and performs color space conversion, resizing, and mean subtraction for preprocessing. The preprocessed frame is then passed through the model to obtain predictions, which are appended to a deque (a double-ended queue) of maximum length 128. The predictions are then averaged over the last 128 frames to obtain the final prediction, and the class with the highest average prediction is chosen as the final result. If the final result is "stain", the code applies background subtraction and thresholding to the frame to detect regions of stains, which are then highlighted with rectangles.
Results:
The detection of stain in fabric gave us a good results it still gets some error w fault positive because the model is trained in few data images while we tested on a video but the performance is good and could be better
6-images : contains the dataset to train the model 
6-1-defect_free: contains data that's free from stains
6-2-stain : contains data that present stains in fabric
7-test_methods:videos test represent the results for eash method