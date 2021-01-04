-------

##### Trained models with weights, (Only Conv_Blocks and Final_Classifier)
- https://github.com/syedfquadri/wgz/releases


- Install all the packages from "Requirements.txt"
## Inference with Trained Models:
> python infer.py -h
```
Inference through trained embedding models(SqueezeNet, Resnet50, EfficientNet B3, and AlexNet) using WGZ food images 
optional arguments:
-h, --help            show this help message and exit
--p1 P1, --img1_path P1
                        Enter path to Image 1. Default: ex_imgs/a1.jpg
--p2 P2, --img2_path P2
                        Enter path to Image 2. Default: ex_imgs/b1.jpg
--emb EMB, --embnet EMB
                        Preferred type of trained emdeddings generator neural network. Default: SqueezeNet[sqe]. Options: [alex,sqe,effB3,resnet]      
--m M, --model M      Preferred model/architecture on which the chosen embedding network has been trained on. Default: Triplet[trip]. Options:       
                        [trip,siam]
--use_gpu             turn on flag to use GPU
```

### Distance between two images
Run "python infer.py" to infer with default images and embedding networks. _SQE_ with _Trip_ works best.

### Distance between images from two directories, Output: a .txt file
- _To be Updated_

## Generate Embeddings from Trained Models of Food Images:
- Three steps to generate embeddings
    - Have a 'data' dir with images in it, classified within classes/labels sub-directory. Shown Below.
    - Generate images Json files using "JsonGen.py", saved in JSONFiles dir.
    - Run "embGen.py"
- The generated projections will be saved as part of a tensorboard runs, in tensorboard "runs" directory. 
```
# _Dirs Structure_
- data
    - 1
        - _____.jpg
        - _____.jpg
    - 2
    - 3
    ...
    - n
```
------------------------------
## Additional Info

### Tensorboard Experiment Links
- *OLD* : https://tensorboard.dev/experiment/XaWjqJX0Sgamt37dAsrTMg
- *NEW* : https://tensorboard.dev/experiment/CgW5mHnqQMqBjT9Imo6iZQ/
### Experiment Types/Approaches
- Final Conv and FC classification Layers (learnable) with Siamese and Triplet Architectures. {*Done*}
- Concating LPIPS_like features extracted from [low, mid and higher] layers with Final Classifier {_*Next Step*_}
- Lottery Ticket Hyporthesis on pretrained net for computer vision tasks. {_*Next Step*_}

### Modules and Classes init
- network
    - AlexEmbNet
    - ResnetEmbNet
    - SQEEmbNet.
    - EfficientNet/det
    - ~~ResNextEmbNet~~
    - SiameseNet
    - TripletNet
- custom datasets
    - SiamDset
    - TripDset
- losses
    - Constrastive Loss
    - Triplet Loss
- train
    -main
- infer
- Embeddings Generator and Projection

### Hyperparameters
- Learning Rate = [~~.01,~~ .001, .0001]
- Batch Size = [ 10 ]
- Contrastive and Triplet Losses Margin = [~~0.2, 0.3,~~ 1.]
- Model = ['siameseNet', 'tripletNet']
- Network = ['sqe', 'alex', 'resnet', 'effB0','effB3'] 
- ~lpips_like = [True, False] {_*Next Step*_}