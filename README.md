##  <div align="center"> Predicting Future Occupancy Grids in Dynamic Environment with Spatio-Temporal Learning  </div>                      

##  <div align="center"> <img src="https://github.com/ksm26/SpatioTemporal-Predictions/blob/master/images/Inria.jpg" alt="INRIA" width="200" height="50"/> </div>


<h3 align="center" id="heading">
    Abstract
</h3>

<p align="justify" width="800">
Reliably predicting future occupancy of highly dynamic urban environments is an important precursor for safe autonomous navigation. Common challenges in the prediction include forecasting the relative position of other vehicles, modelling the dynamics of vehicles subjected to different traffic conditions, and vanishing surrounding objects. To tackle these challenges, we propose a spatio-temporal prediction network pipeline that takes the past information from the environment and semantic labels separately for generating future occupancy predictions. Compared to the current SOTA, our approach predicts occupancy for a longer horizon of 3 seconds and in a relatively complex environment from the nuScenes dataset. Our experimental results demonstrate the ability of spatio-temporal networks to understand scene dynamics without the need for HD-Maps and explicit modeling dynamic objects. We publicly release our occupancy grid dataset based on nuScenes to support further research. 
</p>


<a align="justify" href="url">
  <img src="https://user-images.githubusercontent.com/24546547/177139170-bfd37bd5-9324-4392-b565-faad2138b19e.png" align="center" height="380">
</a>


<h3 align="center" id="heading">
    Generating Occupancy grids dataset
</h3>

<p align="justify">
Occupancy grid maps are generated from the <a href="https://arxiv.org/pdf/1903.11027.pdf">nuScenes dataset</a>. Agents belonging to the ’Vehicles’ category are of interest and being marked by ’green’ semantic pixel labels using the projections of ground truth 3D bounding boxes. Objects of any other type including the static environment are marked in ’blue’. Different road crossing motion scenarios from the dataset are presented below: 
</p>

<a href="url">
  <img src="https://user-images.githubusercontent.com/24546547/177139351-ae4486d2-c493-434f-bba2-bb63fcec0c82.png" align="center" height="180"  width="240">
  <img src="https://user-images.githubusercontent.com/24546547/177139372-3db4e24e-f8a4-4380-aa39-7849b92617d7.png" align="center" height="180"  width="240">
  <img src="https://user-images.githubusercontent.com/24546547/177139409-bb56e1f3-26ed-4d6a-a30a-814718f96659.png" align="center" height="180"  width="240">
  <img src="https://user-images.githubusercontent.com/24546547/177139431-a9630317-3c7f-4164-ad70-66e657d2e73f.png" align="center" height="180"  width="240">
</a>

<p align="justify">
We publicly release the Occupancy Grid Maps dataset consisting of static environment and semantic labels for ease in long-term prediction. The dataset is available <a href="https://archive.org/details/nuscenes-occupancy-grids-dataset">here</a> 

The paper demonstrate the performance of two state-of-art video prediction networks - <a href="https://arxiv.org/pdf/2103.09504.pdf">PredRNN</a> and <a href="https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf">ConvLSTM</a> for this dataset. Under the training scheme we observe that the models have consistent performance even for long-term predictions.
</p>

<h3 align="center" id="heading">
    Video
</h3>

<p align="justify">
<iframe width="560" height="315" src="https://www.youtube.com/embed/4W7dT-HfQPQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>


<h3 align="center" id="heading">
    Spatio-temporal architecture
</h3>

<p align="justify">
We present an Spatio-Temporal Network Pipeline for long-term future occupancy grid prediction. Our approach uses semantic labels of the vehicle in OGMs to model the specific motion type of agents in the prediction rather than using a generic combined prediction of static and dynamic cells that contains the environment and various types of agents such as cars, pedestrians, cyclists, etc.

Semantic occupancy grids consisting of environment and vehicles over the time frames of 0.5 sec. Grids are converted into binary images and separately fed to spatio-temporal networks. We evaluate two spatio-temporal networks: PredRNN and ConvLSTM
</p>
 
![architecture](https://user-images.githubusercontent.com/24546547/177139739-ebd21b21-4644-48a7-bee2-97b32dc8c3d8.png)


<h3 align="center" id="heading">
    Qualitative Results
</h3>

<p align="justify">
A scene depicting few static vehicles and a vehicle going in reverse direction. Comparison between two spatio-temporal learning networks over the future predictions of 1 sec, 2 sec and 3 sec.
</p>

<a href="url">
  <img src="https://user-images.githubusercontent.com/24546547/177150383-e4e77415-1f9e-4ed8-8932-4ffb55ff3e59.jpg" align="center" height="180">
</a>
(a) Ground truth 

<a href="url">
  <img src="https://user-images.githubusercontent.com/24546547/177150412-cc8c02b8-4ae9-40a7-9726-76b611254ed3.jpg" align="center" height="180">
</a>
(b) PredRNN predicitons

<a href="url">
  <img src="https://user-images.githubusercontent.com/24546547/177150364-58998cf4-04a4-44d9-b709-3df36412cd94.jpg" align="center" height="180">
</a>
(c) ConvLSTM predictions


<h3 align="center" id="heading">
    Quantitative Results
</h3>

<p align="justify">
Frame-wise PSNR(↑), SSIM(↑), Static MSE (↓), and Semantic MSE (↓) results on the generated Occupancy grid map dataset are presented below. The prediction horizon is 3 sec during training and testing phases. Note that PredRNN (combined) and ConvLSTM (combined) predict an entire OGM, thus separate Static and Semantic MSE cannot be reported for these cases. The Semantic MSE result from the <a href="https://hal.inria.fr/hal-03416222/document">linear projection</a> of vehicle’s bounding boxes is also presented in (d).
</p>

<a href="url">
  <img src="https://user-images.githubusercontent.com/24546547/177152101-6ed53611-84f5-4290-9853-086f9290a7c8.jpg" align="center" height="170">
  <img src="https://user-images.githubusercontent.com/24546547/177152127-6526887c-4bb3-4305-8116-d418359aa13d.jpg" align="center" height="170">
</a>

<a href="url">
  <img src="https://user-images.githubusercontent.com/24546547/177152144-99bd9fc0-bae6-4edd-b56c-4f3ea9f26732.jpg" align="center" height="170">
  <img src="https://user-images.githubusercontent.com/24546547/177152178-636a03b6-b636-429d-a3f9-c3fa5d377df9.jpg" align="center" height="170">
</a>


<h3 align="center" id="heading">
    Bibtex
</h3>


<p align="justify">
@article{mann2022predicting,
  title={Predicting Future Occupancy Grids in Dynamic Environment with Spatio-Temporal Learning},
  author={Mann, Khushdeep Singh and Tomy, Abhishek and Paigwar, Anshul and Renzaglia, Alessandro and Laugier, Christian},
  journal={arXiv preprint arXiv:2205.03212},
  year={2022}
}
</p>
