# :rocket: GREAT-Stereo (ICCV 2025) :rocket:
This repository contains the source code for our paper.

[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Global_Regulation_and_Excitation_via_Attention_Tuning_for_Stereo_Matching_ICCV_2025_paper.pdf) | [Paper Page](https://iccv.thecvf.com/virtual/2025/poster/1430) | [YouTube](https://youtu.be/ODBqYQIXTBA)

**Global Regulation and Excitation via Attention Tuning for Stereo Matching (GREAT-Stereo)**
<a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Global_Regulation_and_Excitation_via_Attention_Tuning_for_Stereo_Matching_ICCV_2025_paper.pdf">
  <img src="https://img.shields.io/badge/arXiv-2509.15891-b31b1b?logo=arxiv" alt='arxiv'>
</a>

Jiahao LI, Xinhong Chen, Zhengmin JIANG, Qian Zhou, Yung-Hui Li, Jianping Wang

<p align="center"></p>
<div align="center">
  <img src="demos/imgs/architecture.jpg" width="100%" alt="architecture"></img>
</div>
<p></p>

## :bulb: Abstract
Stereo matching achieves significant progress with iterative algorithms like RAFT-Stereo and IGEV-Stereo. However, these methods struggle in ill-posed regions with occlusions, textureless, or repetitive patterns, due to a lack of global context and geometric information for effective iterative refinement. To enable the existing iterative approaches to incorporate global context, we propose the **G**lobal **R**egulation and **E**xcitation via **A**ttention **T**uning (**GREAT**) framework which encompasses three attention modules. Specifically, Spatial Attention (SA) captures the global context within the spatial dimension, Matching Attention (MA) extracts global context along epipolar lines, and Volume Attention (VA) works in conjunction with SA and MA to construct a more robust cost-volume excited by global context and geometric details. To verify the universality and effectiveness of this framework, we integrate it into several representative iterative stereo-matching methods and validate it through extensive experiments, collectively denoted as GREAT-Stereo. This framework demonstrates superior performance in challenging ill-posed regions. Applied to IGEV-Stereo, among all published methods, our GREAT-IGEV ranks first on the Scene Flow test set, KITTI 2015, and ETH3D leaderboards, and achieves second on the Middlebury benchmark.

**Our main contributions are:**
- We propose a universal framework that can be integrated into existing iterative stereo-matching methods to improve the performance in ill-posed regions.
- We introduce Spatial (SA), Matching (MA), and Volume (VA) Attentions, designed to mitigate ambiguities in ill-posed regions with global context information.
- Our method outperforms existing published methods on public leaderboards such as SceneFlow, KITTI, ETH3D, and Middlebury, with especially significant improvements in ill-posed regions.

## :white_check_mark: To Do List
- [ ] The real-time version of the GREAT Framwork.
- [ ] The gpu-memory-friendly implementation of the Matching Attention.
- [x] The Foundation-Model-based experiments.
- [x] The solid and robust version of the GREAT Framwork.
- [x] The accelerate training and evaluating pipeline. 

## :new: Solid Version of GREAT-Stereo
We now propose a solid and robust version of our GREAT Framework, which obtains better performance on the SceneFlow and public KITTI 2012/2015 benchmarks, especially in ill-posed regions like Occlusion. Meanwhile, the Foundation-Model version of our GREAT-IGEV also obtains comparable performance with the current SOTA Foundation-Model-based architectures.

We merge the solid and robust version of GREAT-Stereo into great-stereo folder.

**Our main modifications are:**
- We simplify the implementation of Volume Attention.
- We extend the application of Spatial Attention.
- We remove the redundant implementation of receptive augmentation.
- We modify the cost volume construction pipeline with combined cost volume.
- We implement Foundation-Model (DepthAny) based GREAT-IGEV named GREAT-IGEV-DepthAny by replacing the mobilenetv2 backbone with DepthAnythingV2, which is based on the implementation in [Monster](https://github.com/Junda24/MonSter), and conduct the Foundation-Model-based experiments.
- We accelerate the training and evaluation with DistributedDataParallel settings.

**The benchmark results and corresponding checkpoints are:**
<p align="center"></p>
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px; vertical-align: middle;" rowspan="2">Models</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;" colspan="6">SceneFlow</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;" colspan="4">KITTI2012</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;" colspan="4">KITTI2015</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px; vertical-align: middle;" rowspan="2"">Params</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px; vertical-align: middle;" rowspan="2"">Run Time</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px; vertical-align: middle;" rowspan="2"">Checkpoints</th>
  </tr>
  <tr>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">EPE</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">D3</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Occ-EPE</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Occ-D3</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Non-Occ-EPE</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Non-Occ-D3</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Out-Noc (2px)</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Out-All (2px)</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Out-Noc (3px)</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Out-All (3px)</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">D1-All</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">D1-bg</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Noc-D1-All</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">Noc-D1-bg</td>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;" colspan="18">Light-Weight Model</th>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">LEA-Stereo</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.78</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.90</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.39</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.13</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.45</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.65</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.40</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.51</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.29</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.81M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.30s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">ACVNet</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.48</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.83</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.34</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.13</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.47</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.65</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.37</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.52</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.26</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">6.20M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.20s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">IGEV-Stereo</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.48</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.65</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.19</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.71</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.17</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.12</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.44</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.59</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.38</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.49</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.27</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">12.60M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.32s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">Selective-IGEV</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.45</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.57</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.17</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.59</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.05</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.07</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.38</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.55</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.33</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.44</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.22</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">13.14M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.24s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">IGEV++</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.43</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.56</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.03</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.04</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.36</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.51</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.31</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.42</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.20</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">14.53M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.28s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">GREAT-IGEV (Ours)</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.41</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.20</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.51</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">10.12</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.14</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.49</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.51</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.00</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.02</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.37</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.50</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.28</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">1.37</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">1.14</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">14.44M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.33s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;"><a href="https://drive.google.com/drive/folders/1EZ_hScBixV9opzX7W3ItJXlqS5uNKZvJ?usp=drive_link">Google Drive</a></td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">GREAT-Selective (Ours)</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.42</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.19</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.52</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">10.11</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.15</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.48</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.48</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.94</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.00</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.31</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.49</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.27</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.40</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.16</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">14.98M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.43s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;"><a href="https://drive.google.com/drive/folders/1EZ_hScBixV9opzX7W3ItJXlqS5uNKZvJ?usp=drive_link">Google Drive</a></td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">GREAT-IGEV-Solid (Ours)</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.39</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.13</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.48</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">9.08</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.12</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.47</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.47</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.98</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.95</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.32</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.47</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.25</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">1.37</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">1.14</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">18.4M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.33s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;"><a href="https://drive.google.com/drive/folders/1EZ_hScBixV9opzX7W3ItJXlqS5uNKZvJ?usp=drive_link">Google Drive</a></td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">GREAT-Selective-Solid (Ours)</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.38</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">2.07</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.46</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">8.85</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.11</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.46</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">18.9M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.43s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;"><a href="https://drive.google.com/drive/folders/1EZ_hScBixV9opzX7W3ItJXlqS5uNKZvJ?usp=drive_link">Google Drive</a></td>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;" colspan="18">Foundation Model</th>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">ViTA-Stereo</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.34</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.46</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.80</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.93</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.16</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.50</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.21</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.41</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.12</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">AIO-Stereo</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.58</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.94</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.05</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.29</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.54</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.34</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.43</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.22</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">Foundation-Stereo</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.34</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">DEFOM-Stereo</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.42</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.43</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.79</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.94</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.18</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.41</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.25</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">1.33</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.15</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.30s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">IGEV++ (DepthAny)</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">-</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.36</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.74</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.89</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.13</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.43</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.15</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.36</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.07</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">348M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.48s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">Monster</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.37</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">2.00</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.35</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">9.18</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.14</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.44</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.36</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.75</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.84</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.09</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.41</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.13</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">1.33</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px; font-weight: bold">1.05</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">388M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.45s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">-</td>
  </tr>
  <tr>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">GREAT-IGEV-DepthAny (Ours)</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.36</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">2.03</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.41</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">8.70</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">0.11</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.45</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px; font-weight: bold">1.34</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.76</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">0.85</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.13</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.43</td>
    <td style="border: 1px solid #333; text-align: center; padding: 8px;">1.15</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.36</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">1.07</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">386M</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;">0.43s</td>
    <td style="border: 2px solid #333; text-align: center; padding: 8px;"><a href="https://drive.google.com/drive/folders/1EZ_hScBixV9opzX7W3ItJXlqS5uNKZvJ?usp=drive_link">Google Drive</a></td>
  </tr>
</table>
<p></p>

**The zero-shot results for Foundation Models are:**
<p align="center"></p>
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">Models</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">SceneFlow (EPE)</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">KITTI2012 (D3)</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">KITTI2015 (D3)</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">Middlebury (D2)</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">ETH3D (D1)</th>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">StereoAnywhere</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">-</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.90</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.93</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">6.96</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">1.66</th>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">FoundationStereo</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">0.34</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">-</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">-</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">5.5</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">1.8</th>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">DEFOM-Stereo</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">0.42</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.76</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">4.99</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">5.91</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">2.35</th>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">Monster</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">0.38</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.37</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.44</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.67</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">1.10</th>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">Monster*</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">0.39</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">4.82</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">5.98</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">4.66</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">9.15</th>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">GREAT-IGEV-DepthAny (Ours)</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">0.36</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">4.31</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">5.48</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.35</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">5.82</th>
  </tr>
  <tr>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">GREAT-IGEV-DepthAny* (Ours)</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">0.39</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">4.34</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">5.56</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">3.26</th>
    <th style="border: 2px solid #333; text-align: center; padding: 8px;">2.48</th>
  </tr>
</table>
<p></p>

PS: `Monster*` is the result from the SceneFlow reproduction experiment by using the official code of Monster, see [issue#28](https://github.com/Junda24/MonSter/issues/28) in the official code for more information.

PS: `GREAT-IGEV-DepthAny*` is the result from the SceneFlow experiment after zero-shot selection, according to the [issue#23](https://github.com/Junda24/MonSter/issues/23) in the offcicial code of Monster.

## :clapper: Demos & Results
<p align="center"></p>
<table align="center" width="100%" style="border-collapse: collapse; margin: 20px 0;">
  <tr>
    <td align="center" width="33%">
      <img src="demos/videos/raft.gif" alt="RAFT_DEMO"></img>
      <div style="margin-top: 8px; font-weight: bold;">RAFT Demo</div>
    </td>
    <td align="center" width="33%">
      <img src="demos/videos/igev.gif" alt="IGEV_DEMO"></img>
      <div style="margin-top: 8px; font-weight: bold;">IGEV Demo</div>
    </td>
    <td align="center" width="33%">
      <img src="demos/videos/selective.gif" alt="SELECTIVE_DEMO"></img>
      <div style="margin-top: 8px; font-weight: bold;">Selective Demo</div>
    </td>
  </tr>
</table>
<p></p>

<p align="center"></p>
<div align="center">
  <img src="demos/imgs/sceneflow_vis.jpg" width="100%" alt="sceneflow_vis"></img>
</div>
<p></p>

Qualitative results of GREAT-IGEV on the Scene Flow test set of occlusion (**Row 1**), textureless (**Row 2**), and repetitive texture (**Row 3**) regions.

<p align="center"></p>
<div align="center">
  <img src="demos/imgs/sota_comparison.png" width="100%" alt="sota_comparison"></img>
</div>
<div align="center">
  <img src="demos/imgs/sceneflow_result.jpg" width="100%" alt="transferability"></img>
</div>
<p></p>

Comparisons with state-of-the-art stereo methods on different public benchmarks and ablation study of the cross-model transferability of the proposed GREAT framework on the Scene Flow test set.

## :gear: Environment Settings

* NVIDIA RTX 3090 or 4090
* python 3.8
  
```Shell
conda create -n great python=3.8
conda activate great

pip install torch torchvision torchaudio xformers==0.0.22.post3+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm==4.67.1
pip install scipy==1.10.1
pip install opencv-python==4.11.0.86
pip install scikit-image==0.21.0
pip install tensorboard==2.12.0
pip install matplotlib==3.7.5
pip install timm==0.5.4
pip install numpy==1.24.1
pip install einops==0.8.1
pip install open3d==0.19.0
```

## :floppy_disk: Required Data

* [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [ETH3D](https://www.eth3d.net/datasets)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [TartanAir](https://github.com/castacks/tartanair_tools)
* [VKITTI2](https://europe.naverlabs.com/research/proxy-virtual-worlds/)
* [CREStereo Dataset](https://github.com/megvii-research/CREStereo)
* [FallingThings](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
* [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
* [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)

## :test_tube: Evaluation

1. Download the Checkpoints from [Google Drive](https://drive.google.com/drive/folders/1EZ_hScBixV9opzX7W3ItJXlqS5uNKZvJ?usp=drive_link).

2. Change the following parameters in the script located at `launchers/stereo_matching/test_launcher/`.
    - `dataset`
      - Choices => [sceneflow, kitti, eth3d, middlebury_(Q | H | F)]
    - `dataset_root`
      - your/path/to/corresponding/dataset
    - `restore_ckpt`
      - your/path/to/checkpoint
    - `max_disp` (Optional)
      - `768` for Middlebury and `192` for others

3. Run the evaluation (e.g. Evaluation of GREAT-IGEV on Scene Flow test set).
```Shell
./launchers/stereo_matching/test_launcher/great_igev_evaluator.sh
```

4. (Optional) You can also change the `eval_mode` in the evaluation script to get different evaluation results.
    - `metric` to generate evaluation quantity results (Default).
    - `pcgen` to generate the points cloud of predicted disparity for visualization.
    - `cvvis` to generate the visualization of the cost volume.

## :books: Training

1. Change the following parameters in the script located at `launchers/stereo_matching/train_launcher/`.
    - `logdir`
      - your/path/to/save/training/information
    - `train_datasets`
      - Choices => [sceneflow, vkitti2, kitti, eth3d_train, eth3d_finetune, middlebury_train, middlebury_finetune]
    - `train_datasets_root`
      - your/path/to/corresponding/dataset
    - `restore_ckpt` (Optional)
      - your/path/to/checkpoint/for/finetuning

2. Run the training (e.g. Training of GREAT-IGEV on Scene Flow test set).
```Shell
./launchers/stereo_matching/train_launcher/great_igev_trainer.sh
```

3. (Optional) You can also change the trainer in the script from `stereo_trainer.py` to `stereo_resumable_trainer.py`, which can resume the training if the training process has been accidentally shut down. The `stereo_resumable_trainer.py` will save checkpoints for model, optimizer, and learning rate scheduler for resuming.

4. (Optional) Thanks for the repository of [IGEV-Stereo](https://github.com/gangweix/IGEV/tree/main), we also provide the choices of the data type in mixed precision training. You can change this data type with `precision_dtype` in the script. Choices are `float32`, `float16`, and `bfloat16`. Default value is `float16`. 
**__NOTE__: Our provided checkpoints are trained with `float16` and `float32`.**

## :package: Submission

For submission to the KITTI benchmark (e.g. GREAT-IGEV).
```Shell
python3 save_disp_kitti.py --name great-igev-stereo --restore_ckpt your/path/to/checkpoint --left_imgs your/path/to/left/imgs --right_imgs your/path/to/right/imgs --output_directory your/path/to/save/submission/results
```

For submission to the ETH3D benchmark (e.g. GREAT-IGEV).
```Shell
python3 save_disp_eth3d.py --name great-igev-stereo --restore_ckpt your/path/to/checkpoint --left_imgs your/path/to/left/imgs --right_imgs your/path/to/right/imgs --output_directory your/path/to/save/submission/results
```

For submission to the Middlebury benchmark (e.g. GREAT-IGEV).
```Shell
python3 save_disp_middlebury.py --name great-igev-stereo --restore_ckpt your/path/to/checkpoint --left_imgs your/path/to/left/imgs --right_imgs your/path/to/right/imgs --output_directory your/path/to/save/submission/results
```

## :open_book: Citation

If you find our works useful in your research, please consider citing our paper.
```bibtex
@inproceedings{li2025global,
  title={Global regulation and excitation via attention tuning for stereo matching},
  author={Li, Jiahao and Chen, Xinhong and Jiang, Zhengmin and Zhou, Qian and Li, Yung-Hui and Wang, Jianping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={25539--25549},
  year={2025}
}
```

## Acknowledgements
This project is based on [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [IGEV-Stereo](https://github.com/gangweix/IGEV), and [Selective-Stereo](https://github.com/Windsrain/Selective-Stereo). Meanwhile, the core attention modules of this project are modified from [CoEx](https://github.com/antabangun/coex), [VOLO](https://github.com/sail-sg/volo), and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We thank the original authors for their excellent work.
