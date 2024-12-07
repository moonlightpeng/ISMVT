# ISMVT
we developed a visual inspection system based on deep learning. More precisely, the calculation of gap spacing utilizes the line segment detector of DeepLSD, combining traditional and deep learning techniques to make use of both advantages.

# 1 Notice
1.1 According to different scenarios, camera calibration needs to be completed. Furthermore, The link to the camera used in development is https://item.jd.com/100029997968.html.

1.2 Depending on different inspection products, the thresholds in image processing need to be set according to actual situations.

1.3 The environment for deep learning is configured the same as that for DeepLSD: https://github.com/cvg/DeepLSD.

# 2 Install third party libraries

 pyyaml
 
torch>=1.12

torchvision>=0.13

numpy

matplotlib

brewer2mpl

opencv-python

opencv-contrib-python

tensorboard

omegaconf

tqdm

future  # tensorboard dependency

kornia>=0.6

cython

shapely

scikit-image

h5py

flow_vis

jupyter

seaborn


# 3 Usage

3.1 For camera calibration, you can directly use "CaluateDPI.py". Then the result will be save in the file named wellConfig.xml. For example, the DPI value is 0.0022680113958591258 in current study.

3.2 For Calculation of thresholds in image preprocessing, you can directly use "ButtonValue.py". Then the result will be also save in the file named wellConfig.xml For example, the threshold value is 28 in current study.

3.3 Users can revise the code according to the different applicatons.

3.4 The inspection system can run by "MeasureClient.py" as follows.

![02](https://github.com/user-attachments/assets/f910cc15-4244-47fe-89f8-22ac3ffd552a)

Then we can complete camera calibration using the user interface as follows.
![相机标定](https://github.com/user-attachments/assets/03f23467-30fc-48f8-aa71-adcaa2fd8908)

Additonally, we can perform the Calculation of thresholds in image preprocessing using the user interface as follows.
![04](https://github.com/user-attachments/assets/9e24e351-f6d9-4d2b-a9eb-a9e911be57ca)

Finally, we can perform the inspection of gap spacing by the proposed approach as follows.

![02](https://github.com/user-attachments/assets/522e0a33-893d-4e72-8354-806927a94449)

# 4 If there are any questions about the code, please contact us by ilovemymfandb@163.com.












 
