# CompenHR
## Introduction
This is PyTorch implementation of the paper“[CompenHR: Efficient Full Compensation for High-resolution Projector](http://arxiv.org/abs/2311.13409)”.

## Datasets
The high resolution compensation datasets:

[lavender](https://drive.google.com/file/d/1QA-vllN2RwV_bZOtBV7wYyG4CGyDwmpn/view?usp=sharing), 
[bubble](https://drive.google.com/file/d/1zWalGpOGz2vzsYz1njHBGyiewiMh3Por/view?usp=sharing), 
[cube](https://drive.google.com/file/d/1um2sTthxT-3h1UNgfuoi3zpx93XrtgV4/view?usp=sharing), 
[cloud](https://drive.google.com/file/d/1eBVzFfYCo2KotwvXL0TrmZeZBfN0-rcu/view?usp=drive_link),
[stripes](https://drive.google.com/file/d/15g3UJKamldpWxdGupxUsqPuVMuIPZeAd/view?usp=sharing),
[water](https://drive.google.com/file/d/1b1BgHos_Vz6ieq2YFXEySgqljL1o9IyO/view?usp=sharing), 
[train,test,ref](https://drive.google.com/file/d/1ZBuVkH3XiBOOB4xZ_9I93SJEc4Q_Aufq/view?usp=sharing).

[leaf_np](https://drive.google.com/file/d/1fE1R4OtfrgXrdVBsjWC4PGAWc16o9hc-/view?usp=drive_link),
[lemon_np](https://drive.google.com/file/d/1dV-ZQ5xIsCdDgdLndgxGQhFvhVWrgpCe/view?usp=sharing), 
[flower_np](https://drive.google.com/file/d/1MW2PizScA_nbvGM8pMlYs9q1G9cZ-q7f/view?usp=drive_link),
[rock_np](https://drive.google.com/file/d/1xS5LTsmD0L_le7mx1yb8OLkbq12rVF_q/view?usp=sharing), 
[stripes_np](https://drive.google.com/file/d/1NgLxoLbS2jYuQ9SnSOVLzfZ-byQceQ98/view?usp=sharing).

## Usage
   1. Clone this repo:
  
     git clone https://github.com/cyxwang/CompenHR
     
     cd CompenHR/src/python

   2. Download dataset and extract to ‘data/’
     
   3. Start visdom by typing:
      
     visdom

   4. Run train_CompenHR.py to produce results:
      
     python train_CompenHR.py

