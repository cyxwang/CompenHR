# CompenHR
## Introduction
This is PyTorch implementation of the paper“[CompenHR: Efficient Full Compensation for High-resolution Projector](http://arxiv.org/abs/2311.13409)”.

## Datasets
Before running the code, please download the high resolution compensation datasets:

[lavender](https://drive.google.com/file/d/1QA-vllN2RwV_bZOtBV7wYyG4CGyDwmpn/view?usp=sharing), [bubble](https://drive.google.com/file/d/1zWalGpOGz2vzsYz1njHBGyiewiMh3Por/view?usp=sharing), [train,test,ref](https://drive.google.com/file/d/1ZBuVkH3XiBOOB4xZ_9I93SJEc4Q_Aufq/view?usp=sharing).

[stripes_np](https://drive.google.com/file/d/1NgLxoLbS2jYuQ9SnSOVLzfZ-byQceQ98/view?usp=sharing).

## Usage
   1. Clone this repo:
  
     git clone https://github.com/cyxwang/CompenHR
     
     cd CompenHR/src/python
     
   2. Start visdom by typing:
      
     visdom

   4. Run train_CompenHR.py to produce results:
      
     python train_CompenHR.py

