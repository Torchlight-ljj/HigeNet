# HigeNet
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)

## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

The AIOPS dataset used in the paper can be download in the repo [AIOPSdataset](https://github.com/Torchlight-ljj/AIOPSdataset).
The required data files should be put into `data/DM_TS/` folder. A demo slice of the ETT data is illustrated in the following figure.

<p align="center">
<img src="./img/data.png" height = "168" alt="" align=center />
<br><br>
<b>Figure 3.</b> An example of the ETT data.
</p>


## Usage

