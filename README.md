# CNN-master
CNN based ensemble classifiers trained on multiple data sets with different color spaces

Scene classification systems have become an integral part of computer vision. Recent developments have seen the use of deep scene networks based on
convolutional neural networks (CNN), trained using millions of images to classify scenes into various categories. This paper proposes the use of one such pre-trained
network to classify specific scene categories. The pre-trained network is combined with the simple classifiers namely, randomforest and extra tree classifiers to classify
scenes into 8 different scene categories. Also, the effect of different color spaces such as RGB, YCbCr, CIEL*a*b* and HSV on the performance of the proposed
CNN based scene classification system is analyzed based on the classification accuracy. In addition to this, various intensity planes extracted from the said color
spaces coupled with color-to-gray image conversion techniques such as weighted average, and singular value decomposition (SVD) are also taken into consideration
and their effects on the performance of the proposed CNN based scene classification system are also analyzed based on the classification accuracy. The experiments
are conducted on the standard Oliva Torralba (OT) scene data set which comprises of 8 classes. The analysis of classification accuracy obtained for the experiments
conducted on OT scene data shows that the different color spaces and the intensity planes extracted from various color spaces and color-to-gray image conversion
techniques do affect the performance of proposed CNN based scene classification system.

Requires: Python 2.7, Caffe, Keras
