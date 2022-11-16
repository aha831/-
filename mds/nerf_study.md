# 神经渲染最新进展与算法：NeRF及其演化
**摘要**

​		基于神经辐射场(NeRF)的场景表征与容积渲染无疑是近两年神经渲染方向的爆点工作之一。在提出后的短短一年左右时间内，NeRF以简洁优美的实现思路吸引了大量学者进行深入和拓展研究。本文主要介绍了NeRF方法的基本思想与实现，分析了该方法的优点和局限，探讨了它在计算加速和可编辑渲染方向的一些最新进展。相信NeRF方法会进一步推进神经渲染的发展。我们将持续关注这一领域，并不定期分享我们的认识。

**一、介绍**

​		传统计算机图形学技术经过几十年发展，主要技术路线已经相对稳定。随着深度学习技术的发展，新兴的神经渲染技术给计算机图形学带来了新的机遇，受到了学界和工业界的广泛关注。在近几年的计算机视觉和图形学顶会上，我们都可以看到各种令人耳目一新的神经渲染应用。神经渲染是深度网络合成图像的各类方法的总称。各类神经渲染的目标是实现图形渲染中建模和渲染的全部或部分的功能。

​		基于神经辐射场(NeRF)的场景建模是近期神经渲染的一个热点方向。自20年8月提出后，该研究得到了广泛关注，成为深度学习应用的爆点。许多学者围绕它进行了深入和拓展。在20和21年的CVPR、NeuIPS等各大AI顶会上，我们可以看到几十、上百篇相关的高水平论文。其中GIRAFFE等更是被认为是最佳会议论文。佐治亚理工的Dellaert教授对于NeRF发展的脉络有一篇非常深入的总结，有兴趣的读者建议跟进阅读：https://dellaert.github.io/NeRF/



图1 NeRF相关主题研究(引自crossminds.ai)

 

本文重点介绍了NeRF的深度学习算法和渲染算法的实现，同时也讨论了对其算法优化和可编辑性改进的一些最新进展。

 

 

**二、NeRF场景表征与图像渲染**

NeRF[1]是一种深度渲染方法，其主要特点是场景隐式表达和图像的容积渲染。NeRF实现了全流程可微，因此可以在深度学习框架上方便地实现场景表征的训练优化。

NeRF的基本思想是将目标场景看作3D的容积，用神经网络隐式表征。沿观察方向投影线对3D容积采样，由表征函数计算色彩特征并投影积分后，该方法就可生成渲染图像。因此，NeRF方法的实现是深度场景表征与容积渲染方法的组合。

 

**1. 深度场景表征**

NeRF用神经辐射场来隐式表征场景的色彩特征。在[1]中，神经辐射场是一个深度神经网络。网络的输入是容积化场景体素点的三维位置坐标和观察相机的二维角度坐标，输出是对应五维坐标体素的色彩密度特征。注意，NeRF网络本身并没有直接呈现场景的内容，而是通过输入坐标间接计算出对应点场景信息。因此NeRF是场景的隐函数表征。

 

图 2 场景的容积表征([1]图2(a),(b))

 

NeRF是一个多层感知器网络。注意其输入并不仅仅是原始的3维体素位置与2维相机角度。为了更好表征场景的细节（高频）内容，作者们提出用各维度的高次谐波作为网络输入，如下：

 

其中3维位置坐标每维度包含10次谐波，又以正弦与余弦两种形式共计60个位置输入；二维视角经坐标转换后有三个输入维度，每个维度包含4次谐波，共计24维视角输入。

 

图3 NeRF网络的MLP结构（[1]图7）

 

**2. 容积渲染**

容积渲染是一种特殊的渲染方式，它将以3维体积保存的数据沿观察方向投影形成二维图像。最广为熟知的例子是医学影像中的CT成像。计算机图形学中容积渲染可通过投影面Ray Marching方法来实现。

 

图4 Ray Marching的实现([7])

 

Ray Marching由四步组成：1.在投影图像上逐像素产生射线Raycast；2.沿射线对容积的体素采样；3.获取/计算体素特性；4.累积体素特性计算投影图像的颜色灰度值。

NeRF渲染实现正是按照上述步骤实现，并通过离散采样，投影积分过程转换为累积求和。由于体素特性由可微的MLP函数表征，整个渲染流程是可微的，从而方便在现代深度学习框架上实现。

NeRF的一个创新是针对场景不透明度分布稀疏的特性采用了二次采样的方法来提升采样效率。NeRF方法在一条投影线上先均匀采样64个体素，计算密度分布。根据分布，NeRF再采样128个体素。像素值由两步采样的体素特征共同累加得到。

 

**3. NeRF渲染的实现**

NeRF渲染首先根据场景不同视角的图像学习到场景表征，再由表征生成任意指定角度的图像。因此NeRF方法的实施可以分为两个步骤：场景表征的学习、场景的投影成像。

场景表征的学习是一个深度模型的训练阶段。训练NeRF的深度MLP模型使得其预测的投影图像与作为Ground Truth(GT)的已知图像一致。训练的代价函数是预测图像与GT图像的均方误差。

 

图5 NeRF的训练([1]图1(c),(d))

训练好的NeRF模型成为场景的3D表征函数。给定观察角度，我们就可以按照1.2介绍的容积渲染步骤产生渲染图像。

 

**4. NeRF的优点与局限**

NeRF方法的神经网络模型是一类以坐标为输入的神经网络（Coordinated based Neural Network, CONN）。CONN并非NeRF首创，在多个领域都能找到其应用。近年来在AI融合科学计算中取得瞩目进展的Physical Informed Neural Network(PINN)正是采用了CONN思路来加速微分方程求解[8]。

同样在神经渲染领域，稍早的工作DeepSDF[9]中提出的Signed Distance Function(SDF)等方法也采用基于坐标输入的神经网络来表征场景中体素是否属于目标物体(occupancy)的情况。NeRF方法在这些基础上创新地整合了场景隐式表达和可微容积渲染，用一个新的思路实现了自然场景的学习与再现。其实验结果也非常令人印象深刻。

值得指出的是从计算特征的角度来看，NeRF方法的网络及实现结构并不复杂，其主体网络MLP的计算以矩阵乘加为主。渲染过程仅需要简单的反复执行MLP网络的前向预测。这种计算特征非常适合现代集成了tensor core等部件的AI加速器如GPU、TPU来实现。

NeRF的整体框架也很灵活，易于扩展。其基于坐标的隐式表征方法也进一步启发了其他学者在光场控制、内容编辑等图形渲染其它方向的创新。

作为一个开创性的工作，NeRF也存在一些局限之处。除了渲染质量外，NeRF主要的两个局限在于计算量巨大和仅适用于静态场景。

许多研究工作正是针对这些问题进行了创新和优化。本文主要分析了针对NeRF加速和可编辑渲染的一些进展。有兴趣的读者还可以在下列网站看到更多基于NeRF的最新工作：

https://github.com/yenchenlin/awesome-NeRF

 

**三、NeRF的加速**

由前述介绍可知，NeRF方法生产生图像时，每个像素都需要200余次MLP深度模型的前向预测。尽管单次计算规模不大，但逐像素计算完成整幅图像渲染的计算量还是很可观的。对分辨率不高的百万像素基本图像，高端商用显卡如V100也仅能实现每分钟1，2帧的渲染速度。因此近期人们提出了许多NeRF计算优化的方法。比如FastNeRF[2]、Baking NeRF[3]和AutoInt[4]等。

FastNeRF的基本思路是预先保存所有NeRF表征函数的输出值，这样渲染时无需深度模型计算，查表即可。但原始NeRF是5D坐标输入，即使每维1024分辨率，也需要保存1024**5=1024T的体素特征。为了使得需要保存的值减少到现代显卡可处理的规模，FastNeRF提出利用场景渲染的特性将NeRF模型分解为体素3D位置和投影2D视角两个表征网络分别计算，再组合形成体素色彩特征(如图6)。这样所需尺度由O(n5)降为O(n3)。通过适当的稀疏压缩，整个场景表征可预存到单张显卡，而渲染速度也提升了3000倍！相似地，Baking NeRF也采用了类似的模型分解和预存思路，此处不再赘述。

 

图6 FastNeRF的模型分解([2]图2)

AutoInt是一种不同的加速思路。它将容积渲染中一条射线的投影看作是定积分，并定义对应的神经网络G代表积分过程，然后对该神经网络G求导得到对应的导数网络D。显然，神经网络G,D具有共同的网络参数。AutoInt首先训练导数网络D，然后将优化的参数代入积分函数网络G。给定投影线的起始点，容积投影可通过计算神经网络函数G在两点的差值来计算，即两次对G的前向计算就确定了投影值。图7很清晰的描述了该方法。

AutoInt是从基本的微积分原理出发提出的新型计算方法，充分利用了学习框架的自动微分能力。它不仅仅适用于容积渲染，相信很快我们会在其它领域发掘出它的潜力。

 

图7 AutoInt流程([4]图2)

**四、基于NeRF的可编辑渲染**

由前述可知，原始的NeRF从多幅不同视角投影图像学习到3D场景的静态表征。因此NeRF方法仅能从已有的固定的场景生成渲染图像，无法直接按照主观意图编辑场景内容。这无疑限制了NeRF方法在虚拟现实或内容交互等应用场景的应用。许多学者也看到了这个局限，从不同角度提出了改进方法。本文将探讨几位德国学者的两个相关的工作：GRAF[5]和GIRRAFE[6]。

GRAF

GRAF(Generative Radiance Fields, 生成辐射场)的基本思想是将NeRF方法与生成模型相结合，使得神经网络函数可以表征包含相机与目标物体的相对变换、物体的shape和appearance编码等输入信息的物体，生成辐射场。GRAF也构造了卷积结构的判别器，从GT图像提取具有相同shape和appearance分布的样本与生成图像做比较，进而训练优化生成网络。训练好的生成网络GRAF就可以用于表征3D物体容积信息。

GRAF输入不同的appearance和shape编码和相对坐标，即可在容积中形成期望物体的表征并产生投影图像。因此GRAF实现了对容积渲染内容的可控或可编辑的能力。

 

图8 GRAF方法框架([5]图2)

GIRAFFE

GRAF实现了单个物体的可控表征。在GRAF的基础上，几位学者又开发了GIRAFFE方法来实现可编辑多物体场景的渲染。GIRAFFE也是一个生成模型，但更加复杂。简而言之，GIRAFFE将场景中每个物体和背景分别用一个生成辐射场模型表达，通过appearance、shape和姿态参数控制各物体形态及姿态，通过组合多个GRAF的输出，实现期望场景的表征。

 

图9 GIRAFFE基本流程([6]图1)

容易想到，GIRRAF方法集成了多个辐射场深度模型，整体规模巨大。为了减少容积渲染的计算量，GIRAFF采用了组合渲染的方法加速。首先通过容积渲染生成低分辨率(16x16)的颜色特征图，在根据低分辨率特征用一个2D的CNN渲染器产生最终分辨率高的渲染图像。文章给出的最终图像分辨率是256x256。

相信后续结合NeRF加速方法，我们能够进一步的提升GIRAFFE图像的分辨率与质量。

 

**小结**

NeRF方法提出了一个新的场景建模与渲染的思路。这一神经渲染的开创性工作算法简洁，又充分利用了现代硬件的计算能力，可以称得上是“大巧不工”。这一工作也为神经渲染开辟了新的研究方向，为神经渲染的进一步发展和实用化提供了基础的方法框架。我们可以看到NeRF已经启发了许多学者探索新的神经建模与渲染方法。在今年的AI和图形学顶会上我们看到了一大批令人印象深刻的针对NeRF的改进或基于NeRF的拓展工作。相信这一热潮还会持续下去，在不远的将来会出现NeRF驱动的神经渲染应用于增强现实、虚拟游戏、电影动画等各类图形渲染的实践。

值得一提的是NeRF网络是一种mesh-free的表征方法，与深度科学计算领域正在蓬勃发展的Pyhsics Informed Neural Networks(PINNs)有着内在的共通性。相信这两类方法在模型表达和优化方法等方面会相互促进，进而推动神经渲染和深度科学计算领域的共同发展。

 

**参考文献**

[1] B. Mildenhall, et. al., NeRF: Representing Scenes as neural Radiance Fields for View Synthesis, arXiv:2003.08934v2.

[2] S.Garbin et.al., FastNeRF: High-Fidelity NeuralRendering at 200FPS, arXiv:2103.10380v2.

[3] P. Hedman, et.al, Baking Neural Radiance Fields for real-Time View Synthesis, arXiv:2103.14645v1.

[4] D. Lindell, J. Martel and G. Wetzstein, AutoInt: Automatic Integration for Fast Neural Volume Rendering, arXiv:2012.01714v2.

[5] K.Schwar, GRAF: Generative Radiance Fields for3D-Aware Image Synthesis, NeuIPS 2020.

[6] M. Niemeyer and A.Geiger，GIRAFFE: RepresentingScenes as Compositional Generative Neural Feature Fields, arXiv:2011.12100v2.

[7] T. Gorkin, Volume Rendering using Graphics Hardware, in CIS 565: GPU Programming and Architecture, https://www.seas.upenn.edu/~cis565.

[8] M.Raissi, P.Perdikaris, G.E.Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, J. Compu. Phy., 378(2), 2019.

[9] J. Park, et.al., DeepSDF: Learning Continuous SignedDistance Functions for Shape Representation, CVPR2019.

 

 

 

 

 

一些改进方向：

1. Mipnerf等为主的精度改进方法

2. Instant-ngp grid_encode等加速方法、Mvsf加速方法

3. Wild_nerf等动态噪音去除与光照一致性方法

4. Nsff、D-nerf等时间动态变化

5. Nerfren反光操作方法, 通过学习镜面反射法向量与切向量来优化镜面重建效果(https://bennyguo.github.io/nerfren/)

6. Nerf++ Mipnerf360等远场景重建方法

7. Nerf in dark 等黑暗场景重建方法(https://bmild.github.io/rawnerf/)

 

​		感觉这些场景表面上是在nerf基础上做扩充，表面上目的是重建不同视角下的图像，但本质来看是这种方法具有一种理解场景的能力，这样才能对镜子反光、时间动态、黑暗环境等场景有好的处理效果，并在此基础上进行高级编辑，比如镜面内容改变、动静态分离或黑暗环境曝光程度修改等

 

8. Hypernerf 应对非刚体动态变化场景下的物体渲染问题

9. Citynerf (https://city-super.github.io/citynerf)、blocknerf大场景重建

10. Pixelberf少图片重建

11. Ibrnerf、mvsnerf模型可扩展性

12. Deblur-Nerf 对输入模糊图像的鲁棒性(https://github.com/limacv/Deblur-NeRF)

 

 

Neulf

Viewformer

 

应用方向：

1. 子弹特效（新视角与动态时间合成）：最重要的一点在于提供动态场景的多视角渲染，可用于降低相关场景视频的制作成本，要点在于提高精度

2. 新视角数据合成，可用于生成自动驾驶训练数据集，克服现实中开车采集数据的弊端，获得任意视角任意位置的路况数据；在此基础上，有几个关键点要注意：

​		a)    第一、道路情况往往比较复杂，会有很多移动物体以及环境变化因素，因此要想办法进行动态物体的去处以及环境因素(光照等)的鲁棒处理，可以参考nerf-w，或者想办法进行动静态的分离，提高nsff的精度，再进一步可以进行动态物体的实例化进行单个物体的动态编辑；

​		b)   第二、真实环境往往不再是初始nerf那样对360度近景物体的重建，要考虑对距离跨度较大场景的重建能力，参考nerf++，mipnerf360等方法，有一种采用ndc坐标表示的方法：把视锥体变换到-1～1的立方体坐标系中，可以参考一下

3. 对于相机内外参以及径向畸变的算法优化，主要用于提高精度

4. 使用grid_encode编码方式提高训练与渲染速度，引入超分中的感知loss等函数提高模型生成的场景一致性（是否可以从超分中获取提高精度的方法？）

5. 逆渲染，从nerf的隐式表达到常用3D渲染表达

6. Image2image 变为image2mesh 或者image2grid等sdf方法

 

看nsff代码

一些idea：

1. Coarse_to_fine两阶段的模型与loss改进，不再使用同一个模型同一个reconstructloss来优化coarse与fine网络，而是分别用coarse网络与proploss来优化第一阶段的weight，然后使用fine网络与reconstructloss优化第二阶段的sigma与color

2. Distortion loss的非grid使用

3. deblur-nerf中dsk模糊显示建模模块的使用

 

 

 

 

一些模型项目链接：

0:00 NeRF https://www.matthewtancik.com/nerf

40:22 NeRF-W https://nerf-w.github.io/

49:02 NSFF https://www.cs.cornell.edu/~zl548/NSFF/

1:03:21 D-NeRF https://www.albertpumarola.com/research/D-NeRF/index.html

1:05:35 Mip-NeRF https://jonbarron.info/mipnerf/

1:19:30 Nerfren https://bennyguo.github.io/nerfren/

1:25:45 NeRF++ https://github.com/Kai-46/nerfplusplus

1:29:12 Mip-NeRF 360 https://jonbarron.info/mipnerf360/

1:42:52 RawNeRF https://bmild.github.io/rawnerf/

 

6:27 hypernerf https://hypernerf.github.io/

13:14 neural scene graphs https://light.princeton.edu/publication/neural-scene-graphs/

21:10 ref-nerf https://dorverbin.github.io/refnerf/

33:00 citynerf https://city-super.github.io/citynerf/

46:19 pixelnerf https://alexyu.net/pixelnerf/

49:21 MVSNeRF https://apchenstu.github.io/mvsnerf/

51:30 IBRNet https://ibrnet.github.io/

1:03:27 NVSF https://lingjie0206.github.io/papers/NSVF/

1:07:02 PlenOctree https://alexyu.net/plenoctrees/

1:16:33 KiloNeRF https://github.com/creiser/kilonerf

1:32:30 NeRF-- https://nerfmm.active.vision/

1:37:31 BARF https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/

1:45:09 NDF https://yilundu.github.io/ndf/

2:00:00 閒聊

 

 

 

 

 

 

 

 

## Some idea：

**一、将grid_embed的思路加入到动态场景重建过程中**

​		探索的基本思路就是尝试思考如何将nsff中的各部分优化内容在grid_embed表示的特征空间中进行实现🤔

​		同样使用grid_embed的方法对某场景下的不同空间位置的点的特征进行表示，从而减轻mlp网络的负担，不同之处是此处某个grid处的特征值是根据时间变化的，考虑到动态场景下一般会有很多不同的时间帧，因此对每个不同时间帧下的特征grid进行优化保存的成本是无法接受的；并且为了合成新时间下的场景，网格处的特征值需要具有时间推理性，这里目前想到的可以有两种处理思路：

​		一是以固定位置的视角来看，认为空间中的网格点处的特征值是可以随便变化的，通过输入不同的时间可以查询到当前时间下不同网格点处的特征值；

​		二是以固定空间元素的视角来看，认为空间场景是由一个个独立的三维原子构成的，虽然空间场景中的投影内容是动态的，但是三维原子的特征是始终不变的，只是其位置的变化导致了投影视角内容的变化，这要求网络能够提供两部分内容，一部分是所有三维原子的静态特征，另一部分是随着时间的变化，不同三维原子的位置运动轨迹；

 

目前的卡点：

1. 看代码确定hash encoding的实现方法，是否具有将xyz信息与time信息直接cat起来进行输入的可能性；以及这种形式的实际效果（做实验确定）

2. 搞清楚原nsff渲染时的新时间视角合成方法，确定其在grid embed方式下的实现形式

 

**二、将深度信息加入到动态场景重建中** 

 	这里的深度信息可以是由ssf方法计算而来的稀疏点云，也可以是RGBD采集的密集点云数据，也可以是mvs方法获得的深度信息



**三、NEX类的位置与角度分离的隐式重建方法**（这个得是multiview）

​	The motivation for using the second network is to ensure that the prediction of the basis functions is independent of the voxel coordinates. This allows to precompute and cache the output of `f(x, y, z)` for all coordinates. Therefore a novel view can be synthesized by just a single forward pass of network `g(v)`, because `f()` does not depend on `v` and we don't need to recompute it. 



**四、kilo的small mlp加速方法**

使用nerf蒸馏



**五、加入直线loss限制（不可行😭）**

1. 根据 t 时刻计算的f_t+1得到transformer到 t+1 时刻的三维点在 t 时刻下的空间位置
2. 将空间点的位置变换到 t+1 时刻下的位置
3. 变换后的新位置，即在 t+1 时刻下的位置，理论上应该在同一条直线上，通过计算直线逼近loss优化 f 的计算精度

**`NSFF的改进方向`**

1. 训练与渲染速度 😭
2. 不支持没看到的场景 😭
3. 长时间动态渲染与**物体快速移动**会造成输出平滑与模糊等细节丢失现象 🤔
4. **视角快速移动**也会产生artifacts（物体快速移动 & 视角快速移动 不一样）





### Scratch


## NSFF专项

### 改进思路

1. distort_loss已加入，但效果仍需调试，尤其是其中interval的给定（w与s较好确认）
2. 已经将deblur中的Deformable Sparse Kernel (DSK) 方法加入到动态场景中，指标明显下降😭
3. 考虑`transirnt_weight自身进行entropy计算`以及`static_weight与thick_filt后的transient_weight计算交叉熵`是否可进一步改进
4. 研究动静态分离方法，考虑如何进行动态场景的背景替换渲染（可能需要提前渲染好背景静态区域的模型）
5. 动态场景的风格迁移











**NVIDIA动态场景数据：**

1. 一共8个场景，每个场景都是由12个位置(固定的)相机进行拍摄的；
2. 也就是说该数据集的每个场景最多就只有12个不同的视角，训练数据中一个视角基本都有多个对应的时间下的场景
3. nsff验证时是用时间上完全不重叠的图片组来验证的，目的是保证能够同时确保测试的时间和视角都不在训练集中（这要求训练集中必须至少避开12个相机中的一个，并在验证的时候使用这些训练时避开的视角）



目前实验结果的问题：

1. 动静态分离效果差：主要是原动态区域部分被识别为静态，说明模型计算的动态与静态权重与实际相差较大，目前识别动静态区域的方式是分别预测动静态权重然后加权（并对动静态场景的峰值距离进行限制），实际上这里就是一种二分类，动静态权重就是在分离时参考的概率大小，只不过在全场景渲染时基于两个权重对动静态颜色进行了加权。
2. 后向光流没有输出：检查一下，可能是代码问题







## 工具

1. 模型框架工具 pytorch-lightning
2. tensor格式转换工具 einops ( rearrange/reduce/repeat )

 

## 内容记录 

### 八叉树Otree 与 morton3D

https://qa.1r1g.com/sf/ask/69012541/

https://zhuanlan.zhihu.com/p/195730581

https://blog.csdn.net/qq_28660035/article/details/80488517

https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/#:~:text=If%20you%20want%20to%20convert,cell%20along%20the%20Z%2Dcurve


BN, LN, IN, GN：(N, C, H, W)

1. bn是针对每一个C，都计算N✖️H✖️W的均值与方差

2. Ln是针对每一个N，都计算C✖️H✖️W的均值与方差，相当于计算每张图片的全通道均值方差
3. GN也针对每一个N，计算C✖️H✖️W的均值与方差，只不过这里的C分成了不同的组来计算(C1✖️H✖️W, C2✖️H✖️W, C=C1+C2)，相当于每次只计算计算每张图片的部分通道的均值方差
4. Ln是针对每一个N,C，都计算H✖️W的均值与方差，相当于计算每张图片的单个通道的均值方差



BN的参数：

1. 因为BN是针对不同的C在N内做计算，所以参数量一定是C的倍数
2. 其中可学习的参数有两个，weight/bias，二者的意义就是避免所有层的特征都被归一化为正态分布而失去特征差异性，因此通过这两个参数来恢复模型对不同层特征的敏感度
3. mean/val则是在训练过程中随着迭代逐步使用动量法更新的，在测试前中通过net.eval()命令来固定



softmax作用：

1. 非负性
2. 和为1
3. 输入大于1的情况就被拉大倍数差距，小于1的情况会被缩小倍数差距(拉近)
4. 多分类时不能用这个



sigmoid：y=1/(1+exp(-x))



二分类时不用mse而用交叉熵的原因：真实值是一种二值阶跃变化，无法很好的用mse拟合