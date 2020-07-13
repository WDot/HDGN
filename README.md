# HDGN
Hard Directional Graph Network

This is a 3D Point Cloud Classification/Segmentation network submitted to ICPR 2020.

All experiments were run on a single 32GB V100.

This code requires the following software:
* Python 3
* H5py 2.9.0+
* Scikit Learn 0.21.3+
* Pytorch

To train the model, go into the ModelNet40 folder and run the following command:

```
python3 main.py --exp_name=train --model=hdgn --num_points=1024 --k=32 --use_sgd=True --device=cuda:0 --epochs=250 --lr=0.001 --num_theta=5 --num_phi=5
```

To evaluate the pretrained model, run the following command:

```
python3 main.py --exp_name=pretrained_eval --model=hdgn --num_points=1024 --k=32 --device=cuda:0 --num_theta=5 --num_phi=5 --eval=True --model_path=checkpoints/pretrained/models/model.t7 --test_batch_size=8
```
