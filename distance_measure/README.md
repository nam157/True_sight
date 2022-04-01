### CÁCH 1: stereo camera
![image](https://github.com/nam157/ai4theblind/blob/main/distance_measure/Images/AnhDoAn.png)
#### Công thức: 
![image](https://github.com/nam157/ai4theblind/blob/main/distance_measure/Images/congthuc.png)




### CÁCH 2: Dispnet

The median value of the estimated distances of all pixels inside the 
bounding box of an object in a depth imageis computed (REV)

However to convert the REV distance to absolute distance (ABS), the real distance 
of objects in images are needed

In most works of ABS estimation, ABS of 
an object depends on the type and shape of objects, as well as the image size and focal 
length of the sensor

we need to avoid depending on this
type of information and we will try to calibrate our method to work for different 
unknown objects.

the ABS distance can be 
calculated based linear regression

#### Results 1:

![image](https://github.com/nam157/ai4theblind/blob/main/distance_measure/Images/ezgif-1-538cc99cb6.gif)


### CÁCH 3: monocular camera
![image](https://user-images.githubusercontent.com/72034584/161086094-8c802bf8-e915-4cc7-b2c3-389bba45c969.png)

Công thức: 

![image](https://user-images.githubusercontent.com/72034584/161089684-14436672-82dc-447e-a14d-c069b37068b7.png)


* focal length (f)
* radius of marker in the image plane (r) 
* radios of marker in the object plane (R) and unknown parameter
* distance from the camera to the object(d).



#### Results 2:
![image](https://github.com/nam157/ai4theblind/blob/main/distance_measure/Images/ezgif-5-f0d2ef8c9d.gif)

### Reference
* https://arxiv.org/ftp/arxiv/papers/2111/2111.01715.pdf
* http://emaraic.com/blog/distance-measurement
* https://www.khanacademy.org/science/physics/geometric-optics/lenses/v/object-image-and-focal-distance-relationship-proof-of-formula
