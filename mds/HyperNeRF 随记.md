## HyperNeRF 随记

1. 对于动态物体的空间重建，一个关键之处是正确推理物体形状的变化，一种常见的方法是**deformation field**，即D-Nerf中使用的形变变换网络。该方法会计算当前时刻物体的场变大小(每个点对应的位置偏移ΔX)，然后加到**canonical space**上得到当前时刻的space。

   ###### 该方法(deformation field)中都有一个canonical space，指的是初始无形变时的空间形状，其实这个space可以是众多已知形变状态中的任何一个，不过一般都将t=0时的初始状态视作这个无形变的canonical space。

   ###### 拓扑变化(切水果、撕开纸张)中会发生物体表面连通性的变化，这要求变形场是不连续的，而deformation field这种变形场中描述的形变都是连续的(也许是因为该方法使用的基本都是fcn等连续函数来实现的？)，会导致三维重建过程中出现奇异点或运动不连续的现象，这就引申出了另一种推理物体形状变化的方法——**level set**。

   ​		

2. **level set**类似一种降维打击，将三维场景作为四维或更高维空间中的一个切片，把运动场景建模成高维空间中的静态物体，拓扑变化就成了光滑的而不是不连续的；

   ​		这些增加的维度则称为**ambient dimensions**，比如4D动态场景中的时间，ambient dimentions可以是大于1的；

   ​		在高维空间中获取三维拓扑变换结果的切片可以是直面**Axis-aligned Slicing Plane** (AP) ， 也可以是通过MLP等模拟的曲面**Deformable Slicing Surface** (DS)

![image-20220917214246480](C:\Users\19596\AppData\Roaming\Typora\typora-user-images\image-20220917214246480.png)

###### 				Deformable Slicing Surface可以让切片之间共享信息，ambient surface会很简单

