# pytorch-cnn-mnist
This repo contains a sample code to show how to create a cnn model using pytorch and trained on the mnist.
## demo
> [http://pytorch-cnn-mnist.herokuapp.com/](http://pytorch-cnn-mnist.herokuapp.com/)

![](https://img-blog.csdnimg.cn/20201116104632753.gif)
## install
Install pytorch and other necessary module.
```bash
pip install -r requirements.txt
```
## train
Run train.py to train your cnn model on mnist dataset.
```bash
python train.py
```
## test
Run test.py to test the trained model on your own handwritten digit.
```bash
python test.py
```
## deploy
1. register an account of heruku
2. create an app on heruku
3. push the `deploy` folder to heroku master
