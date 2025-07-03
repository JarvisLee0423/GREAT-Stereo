# :rocket: GREAT-Stereo (ICCV 2025) :rocket:
This repository contains the source code for our paper:

**Global Regulation and Excitation via Attention Tuning for Stereo Matching (GREAT-Stereo)**
<a href="TBC">
  <img src="https://img.shields.io/badge/arXiv-TBC-b31b1b?logo=arxiv" alt='arxiv'>
</a>

Jiahao LI, Xinhong Chen, Zhengmin JIANG, Qian Zhou, Yung-Hui Li, Jianping Wang

<div style="text-align: center">
  <img src="demos/imgs/architecture.jpg" alt="architecture"></img>
</div>

## :bulb: Abstract
Stereo matching achieves significant progress with iterative algorithms like RAFT-Stereo and IGEV-Stereo. However, these methods struggle in ill-posed regions with occlusions, textureless, or repetitive patterns, due to a lack of global context and geometric information for effective iterative refinement. To enable the existing iterative approaches to incorporate global context, we propose the **G**lobal **R**egulation and **E**xcitation via **A**ttention **T**uning (**GREAT**) framework which encompasses three attention modules. Specifically, Spatial Attention (SA) captures the global context within the spatial dimension, Matching Attention (MA) extracts global context along epipolar lines, and Volume Attention (VA) works in conjunction with SA and MA to construct a more robust cost-volume excited by global context and geometric details. To verify the universality and effectiveness of this framework, we integrate it into several representative iterative stereo-matching methods and validate it through extensive experiments, collectively denoted as GREAT-Stereo. This framework demonstrates superior performance in challenging ill-posed regions. Applied to IGEV-Stereo, among all published methods, our GREAT-IGEV ranks first on the Scene Flow test set, KITTI 2015, and ETH3D leaderboards, and achieves second on the Middlebury benchmark.

## :clapper: Demo & Results:
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

<div style="text-align: center">
  <img src="demos/imgs/sceneflow_vis.jpg" alt="sceneflow_vis"></img>
</div>

Qualitative results of GREAT-IGEV on the Scene Flow test set of occlusion (**Row 1**), textureless (**Row 2**), and repetitive texture (**Row 3**) regions.

<div style="text-align: center">
  <img src="demos/imgs/sota_comparison.jpg" alt="sota_comparison"></img>
</div>
<div style="text-align: center">
  <img src="demos/imgs/sceneflow_result.jpg" alt="transferability"></img>
</div>

Comparisons with state-of-the-art stereo methods on different public benchmarks and ablation study of the cross-model transferability of the proposed GREAT framework on the Scene Flow test set.
