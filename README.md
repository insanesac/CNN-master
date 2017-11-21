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

The first step involved feeding the RGB data set to the pretrained CNN model. The
comparatively small data set made it hard to design a new neural network that could
give satisfactory accuracy and hence a predefined model was chosen. But before the
dataset was fed into the CNN, the data set had to be split into training and testing
set. The RGB data set was split into training and testing set, comprising of 1888 and
800 images respectively. 

The images were then renamed according to their classes, for example, coast1,
coast2 etc., to facilitate easy labeling. The classes were designated with labels from
0 to 7.
Due to the small size of data set, training a CNN for the purpose of classification
is really hard. And hence, the experiment was done using a pretrained network.
For the purpose of training Places-CNN, 2,448,873 images from 205 categories of
Places-205 dataset were selected randomly as the train set, with minimum 5,000
and maximum 15,000 images per category. The validation set was made of 100
images per category while the test set contained 200 images per category to give
a total of 41,000 images. Places-CNN model was trained using the Caffe package
on a NVIDIA Tesla K40 GPU and took about 6 days to finish 300,000 iterations of
training [17]. The network contains eight layerswith weights; five convolutional and
three fully-connected layers. The output of the last fully-connected layer produces
a distribution over the 205 class labels. The input layer for the network is an image
of size 224 × 244 × 3, which is then fed to a convolutional layer with 96 kernels
of size 11 × 11 and a stride of 4 pixels. To the output of every convolutional and
fully connected layer, a ReLU nonlinearity is applied to. Each neuron computes the
weighted sum of its inputs and applies an offset which then runs the result through
a nonlinear function, ReLU.

The pretrained model predicted, out of the 205 classes, to which class the new
observation belongs to. For the purpose of this experiment, rather than predicting
a particular class, the model was made to give an output of 205 values. This step
was necessary because, while the dataset we used had specific classes, for example,
coast, the 205 classes had classes like coast, pond, aquarium and so on. To simplify
it, an image from the class coast was categorized to coast, pond, aquarium which all
had a common factor, water.
The model was then fed with the training and testing data set separately. The
training set comprised of 1888 images, that means for each image, the model gave
out 205 probabilistic values for each image. These 205 values when added resulted
into 1, in other words, these values were probabilistic values. In a classification task,
generally, the likelihood of a particular image falling into all the classes is calculated
by the CNN, and then the class with the highest probability or likelihood value
is then considered as the class in which the image belongs. Here, since there are
205 classes, the CNN will calculate 205 probability values for every single image.
Instead of choosing the highest probability value, we made use of the entire 205
values to form a feature vector which was then used for further classification. Thus
a matrix of shape 1888 by 205 was obtained, each vector of size 205 represented
one image. The same was done with the testing set to obtain a matrix of shape 800
by 205.


Requires: Python 2.7, Caffe, Keras
