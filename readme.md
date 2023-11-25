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

## How to run

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

## Results

Results are shown in the handed-in report.

## References

The code is based on the following repositories:

- [Micromind](https://github.com/micromind-toolkit/micromind)