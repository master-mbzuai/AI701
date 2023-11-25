# AI701 project fall 2023

## Team members
- [ ] 1. [Farhkhad]()
- [ ] 2. [Arsen]()
- [ ] 3. [Sebastian Cavada]()

## Project description

## Project structure

- pretrained architecture -> report the baseline data with the pretrained weights and pretrained classifier
- original classifier -> fine tune the classifier given the pretrained weights, the classifier is the original, but has to be retrained to fit the new dataset and provide realistic performance with different epochs training
- adaptive classifier -> fine tune the first modification of the classifier, the classifier is now two levels deep and it does not have any non-linearity
- nmf adaptive classifier -> fine tune the second modification of the classifier, the classifier is now two levels deep and it does have non-linearity (at the beginning and at the beginning)
- hierarchy classifier -> fancier stuff with the most interesting results (hopefully)

### Other Folders

- experiments -> all the experiments performed, with the different scripts used
- results -> all the results of the experiments, with the different folders for each experiment
- analysis ->  some helper scripts used to analyze the results
- components -> custom components developed for the project

### Installation 

1. Git clone the repo ```https://github.com/master-mbzuai/micromind.git``` and install it with ```pip -e install .``` while being inside the folder.
2. Run ```pip install -r requirements.txt```

## How to run

Refer to the installation phase

There is a specific order to reproduce our results.

0. Step 0 is to get the pretrained weights either for PhiNet or Resnet (get access at link: https://huggingface.co/micromind/ImageNet)

1. Once the pretrained weights are downloaded or the access was granted, the first step is to run the python script:

```python main.py --model_name original --experiment_name finetuned --epochs 50 --lr 0.0001```

This will fine-tune the backbone on cifar 100. A new folder will be created in the results folder named finetuned. This folder will contain the weights of the backbone and the weights of the classifier.

At this point many different experiments were performed, and all of the script used are found in the /experiments folder. Running either one will perform one of the experiments described.

For the sake of brevity I will guide you here to the most interesting experiments.

2. The second step is to run the python script:

```python main_hierarchy10.py --model_name hierarchy10 --experiment_name hierarchy10 --epochs 50 --lr 0.0001```

This script will take the weights of the backbone and the weights of the classifier from the previous step and will fine-tune a classifier with only 10 classes, which are the parent classes of all the classes in CIFAR100.

3. The third and last step is to run the python script:

```python main_hierarchy100.py --model_name hierarchy100 --experiment_name hierarchy100 --epochs 50 --lr 0.0001```

This will use the previous trained weights and fine-tune the classifier on the whole CIFAR100 dataset, with the hierarchical approach described in the report.

## How to run ResNet
If you want to reproduce the results of experiments with ResNet from scratch, you can do this:
0. you don't need any weights, so no need to download them
1. Run all of the cells in ResNet1028_to_100class.ipynb . This is the baseline model
2. Run all of the cells in ResNet1028_to_d_to_100_plus_graph.ipynb . This is the script for introducing matrix factorization to the ResNet model.
3. Run all of the cells in ResNet1028_to_parent_Loss_parent_only.ipynb . This is where the model is trained to do parent classification only. In the last few cells the weights are stored as 'group_1028_to_parent_Loss_parent_only_pretrained_model.h5'. They're going to be used later.
4. Go to 20_Child_Classifiers/ResNet1028_to_parent_to_child_Loss_parent_only_LOOP copy.ipynb and  run all of the cells. The accuracy of child classification for each of the 20 parents are stored in a list. The arithmetic average of child classification accuracy is calculated in the last cell. The accuracy is suboptimal, so the weigths are not stored.
5. Run all of the cells in ResNet1028_to_parent_Loss_parent_and_child.ipynb. This is where the model is trained to do both parent and child classification (Loss has two components). In the last few cells the weights are stored as 'group_1028_to_parent_Loss_parent_and_child_pretrained_model.h5'. They're going to be used later.
6. Go to 20_Child_Classifiers/ResNet1028_to_parent_to_child_Loss_parent+child_LOOP.ipynb and run all of the cells. Twenty child classifiers are trained to do child classification and their weights are stored in the format 'group_1028_to_parent_Loss_parent_and_child_pretrained_model_For_parent_{i}.h5' where i is ranging from 0 to 19.
7. The problem is that we save redundant information from those models in the previous step. We only need child classifiers' weights. To keep only those weights that we need, go to 20_Child_Classifiers_reduces_size/ResNet1028_to_parent_to_child_Loss_parent+child_LOOP.ipynb and run the cells. It will store weights in the forms of tensors, because in child classification, there is just one simple operation: matrix multiplication and bias being added. For each parent, there is a tensor 'group_1028_to_parent_Loss_parent_and_child_pretrained_model_For_parent_{}_weight_reduced_size.pt' and tensor 'group_1028_to_parent_Loss_parent_and_child_pretrained_model_For_parent_{}_bias_reduced_size.pt'.git
8. Run all of the cells in Creating_json_parent_to_child.ipynb . This will create a json file, describing a mapping between fine label and child label. It's a nested dictionary:
{key: (parent) & value: dictionary {key: (fine label) & value: (child label)}}
9. Run all of the cells in 20_Child_Classifiers_reduces_size/ResNet_1028_to_parent_to_child_hierarchy.ipynb . This is the final result of branching CNN approach of ResNet on the test dataset (100 labels).

If you want to reproduce the results of testing hierarchy with the weights provided from OneDrive, you can do this:
0. Download 'group_1028_to_parent_Loss_parent_and_child_pretrained_model.h5' and run all of the cells in 20_Child_Classifiers_reduces_size/ResNet_1028_to_parent_to_child_hierarchy.ipynb.

## Results

Results are shown in the handed-in report.

## References

The code is based on the following repositories:

- [Micromind](https://github.com/micromind-toolkit/micromind)