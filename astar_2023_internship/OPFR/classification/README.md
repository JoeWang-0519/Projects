# Forward PFH (based on RepSurf) for Classification <br>

By *[Haoxi Ran\*](https://hancyran.github.io/) , Jun Liu, Chengjie Wang* ( * : corresponding contact)

### [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ran_Surface_Representation_for_Point_Clouds_CVPR_2022_paper.pdf) | [arXiv](http://arxiv.org/abs/2205.05740)


## Preparation

### Environment

We tested under the environment:

* python 3.7
* pytorch 1.7.0
* cuda 10.1
* gcc 7.2.0
* h5py
* tqdm

For anaconda user, initialize the conda environment pfh-cls by:

```
sh init.sh
```

## Experiments

### ScanObjectNN

* Performance:

<table style="width:100%">
  <thead>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>#Params</th>
      <th>Augment</th>
      <th>Code</th>
      <th>Log</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/ajhamdi/MVTN">MVTN</a></td>
      <td align="center">82.8</td>
      <td align="center">4.24M</td>
      <td align="center">None</td>
      <td align="center"><a href="https://github.com/ajhamdi/MVTN/blob/master/models/mvtn.py">link</a></td>
      <td align="center">N/A</td>
      <td align="center"><a href="https://github.com/ajhamdi/MVTN/blob/master/results/checkpoints/scanobjectnn/model-00029.pth">link</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/ma-xu/pointMLP-pytorch">PointMLP</a></td>
      <td align="center">85.7</td>
      <td align="center">12.6M</td>
      <td align="center">Scale, Shift</td>
      <td align="center"><a href="https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ScanObjectNN/models/pointmlp.py">link</a></td>
      <td align="center"><a href="https://web.northeastern.edu/smilelab/xuma/pointMLP/checkpoints/fixstd/scanobjectnn/pointMLP-20220204021453/">link</a></td>
      <td align="center"><a href="https://web.northeastern.edu/smilelab/xuma/pointMLP/checkpoints/fixstd/scanobjectnn/pointMLP-20220204021453/">link</a></td>
    </tr>
    <tr>
      <td align="center">PointNet++ SSG</td>
      <td align="center">77.9</td>
      <td align="center">1.475M</td>
      <td align="center">Rotate, Jitter</td>
      <td align="center"><a href="https://github.com/hkust-vgd/scanobjectnn/blob/master/pointnet2/models/pointnet2_cls_ssg.py">link</a></td>
      <td align="center">N/A</td>
      <td align="center">N/A</td>
    </tr>
    <tr>
      <td align="center"><b>Umbrella RepSurf</b> (PointNet++ SSG)</td>
      <td align="center"><b>84.87</b></td>
      <td align="center">1.483M</td>
      <td align="center">None</td>
      <td align="center"><a href="models/repsurf/repsurf_ssg_umb.py">link</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1qJK8T3dhF6177Xla227aXPEeNtyNssLF/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/17UDArfvNVjrJBTjr_HdxcOQipn0DWMMf/view?usp=sharing">google drive (6MB)</a></td>
    </tr>
    <tr>
      <td align="center"><b>Umbrella RepSurf</b> (PointNet++ SSG, 2x)</td>
      <td align="center"><b>86.05</b></td>
      <td align="center">6.806M</td>
      <td align="center">None</td>
      <td align="center"><a href="models/repsurf/repsurf_ssg_umb_2x.py">link</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/15HwmAi1erL68G08dzNQILSipwCIDfNAw/view?usp=sharing">google drive</a></td>
      <td align="center"><a href="https://drive.google.com/file/d/1yGPNt1REzxVwn8Guw-PFHFcwxvfueWgf/view?usp=sharing">google drive (27MB)</a></td>
    </tr>
    <tr>
      <td align="center"><b>Umbrella RepSurf + Forward PFH </b> (PointNet++ SSG)</td>
      <td align="center"><b>85.7</b></td>
      <td align="center">None</td>
      <td align="center">None</td>
      <td align="center"><a href="models/repsurf/repsurf_ssg_pfh.py">link</a></td>
      <td align="center">None</td>
      <td align="center">None</td>
    </tr>
  </tbody>
</table>
<br>

* To download dataset:

```
wget https://download.cs.stanford.edu/orion/scanobjectnn/h5_files.zip
unzip h5_files.zip
ln -s [PATH]/h5_files data/ScanObjectNN
```

**Note**: We conduct all experiments on the hardest variant of ScanObjectNN (**PB_T50_RS**).
<br>

* To train **Umbrella RepSurf + Forward PFH** on ScanObjectNN:

```
sh scripts/scanobjectnn/repsurf_ssg_pfh.sh
```

* To train **Umbrella RepSurf** with **Modified Sampling Shceme** on ScanObjectNN:

```
sh scripts/scanobjectnn/repsurf_ssg_umb_hier.sh
```

* To train **Umbrella RepSurf + Forward PFH** with **Modified Sampling Scheme** on ScanObjectNN:

```
sh scripts/scanobjectnn/repsurf_ssg_pfh_hier.sh
```

* To train **Umbrella RepSurf + Estimate PFH (with Estimate Normal/Dataset Normal)** on ScanObjectNN:

```
sh scripts/scanobjectnn/repsurf_ssg_umb.sh
```
## Remark 1
1. If we want to use **Manual PFH Features**, we need construct PFH features from estimate/dataset normals.

On ScanObjectNN dataset, we only support Estimate Normal. 

On ModelNet40 dataset, we can choose Estimate Normal/Dataset Normal.  

'--use_normals' arg is to dataset/estimate normals for PFH computation. In ModelNet40 dataset, we load dataset normals by default. If we want to load estimate normals, we should use '--estimate_normals' arg.

'--use_pfh' arg is to generate PFH features based on dataset/estimate normals. If we use PFH features, then input data size is [N, 3+C], C is PFH feature channel size (30 by default).

2. If we want to use **Forward PFH Features**, we do not need estimate/dataset normals. Therefore, there is no need for '--use_normals' & '--use_pfh' args.



## Acknowledgment

Coding part is mainly based on **RepSurf**, whose author is **Haoxi Ran (ranhaoxi@gmail.com)**. The github link is attached: https://github.com/hancyran/RepSurf/tree/main/classification.
