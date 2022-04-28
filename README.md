#  <div align="center"> SpatioTemporal-Predictions </div>

##  <div align="center"> Predicting Future Occupancy Grids in Dynamic Environment with Spatio-Temporal Learning  </div>                      

##  <div align="center"> <img src="https://github.com/ksm26/SpatioTemporal-Predictions/blob/master/images/Inria.jpg" alt="INRIA" width="200" height="50"/> </div>


## Abstract
Reliably predicting future occupancy of highly dynamic urban environments is an important precursor for safe autonomous navigation. Common challenges in the prediction include forecasting the relative position of other vehicles, modelling the dynamics of vehicles subjected to different traffic conditions, and vanishing surrounding objects. To tackle these challenges, we propose a spatio-temporal prediction network pipeline that takes the past information from the environment and semantic labels separately for generating future occupancy predictions. Compared to the current SOTA, our approach predicts occupancy for a longer horizon of 3 seconds and in a relatively complex environment from the nuScenes dataset. Our experimental results demonstrate the ability of spatio-temporal networks to understand scene dynamics without the need for HD-Maps and explicit modeling dynamic objects. We publicly release our occupancy grid dataset based on nuScenes to support further research.

##  <div align="center"> <img src="https://github.com/ksm26/SpatioTemporal-Predictions/blob/master/images/teaser.png" alt="teaser" width="800" height="400"/> </div>

## Dataset generation
The dataset is hosted at: <a href="https://archive.org/details/nuscenes-occupancy-grids-dataset">https://archive.org/details/nuscenes-occupancy-grids-dataset</a>

We proceed by generating occupancy grids from NuScenes dataset by superimposing the 3D bounding boxes on the the grids. The generated dataset can be seen as:
 
 ##  <div> <img src="https://github.com/ksm26/SpatioTemporal-Predictions/blob/master/images/scene1.gif" alt="Scene1" width="300" height="200"/> </div>
 ##  <div> <img src="https://github.com/ksm26/SpatioTemporal-Predictions/blob/master/images/scene2.gif" alt="Scene1" width="300" height="200"/> </div>





