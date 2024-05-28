+++
title = "Transfer Learning Classifier Again... with Julia!"
date = 2024-05-21T22:38:29+08:00
draft = false
summary = "Replicating the cat mood classifier, this time using Julia and Flux.jl."
tags = ["Classification", "Deep Learning", "Machine Learning", "Julia", "Flux.jl", "Transfer Learning"]
+++
---

![image](/images/20240521_julia_transfer_learning_v5/intro.png)

## Introduction
This guide demonstrates how to apply transfer learning using a pre-trained vision model to classify cat moods based on their facila expressions. We'll learn how to handle custom data setups.

In this demonstration, we recreate the exercise done in PyTorch, [available here](https://vflores-io.github.io/posts/20240515_cat_mood_classification/). Since that demonstration is quite detailed, we keep it pretty straightforward here.

#### Motivation & Credit

When I thought about learning how to implement a computer vision classification model for transfer learning in Julia and `Flux`, I immediately came upon two roadblocks:
1. Since I am not an expert in Julia, I found the documentation to be a bit difficult to access (again, this is just me!).
2. There are not many tutorials or resources to illustrate this particular case.

Therefore I took it upon myself to put things together and make a demonstration that would hopefully be useful for someone who might not be an expert in Flux (or Julia).

This particular demo was inspired by a combination of the following resources:

- [Transfer Learning and Twin Network for Image Classification using `Flux.jl`](https://towardsdatascience.com/transfer-learning-and-twin-network-for-image-classification-using-flux-jl-cbe012ced146)
- [`Flux.jl`'s Model Zoo Tutorial](https://github.com/FluxML/model-zoo/tree/master/tutorials/transfer_learning)
- [`PyTorch` Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
## Getting Started

We will use a pre-trained `ResNet18` model, initially trained on a general dataset, and fine-tune it for our specific task of classifying cat moods.


### Initialization

First, we activate the current directory as our project environment by calling the package manager `Pkg`:

```julia
using Pkg
Pkg.activate(".") 
```

Then we will import the required packages. Of course, this is also assuming that one has already added the relevant packages into the environment.


```julia
using Pkg
Pkg.activate(".")

using Random: shuffle!
import Base: length, getindex
using Images
using Flux
using Flux: update!
using DataAugmentation
using Metalhead
using MLUtils
using DataFrames, CSV
using Plots
```

    [32m[1m  Activating[22m[39m project at `H:\My Drive\Projects\Coding\Portfolio\Machine Learning\Julia\Transfer Learning with Flux`
    

### Retrieve the Data and Initial Setup

First, we specify the paths to the dataset and labels CSV files for training, validation, and test sets. Then, we load these CSV files into `DataFrames`. Finally, we create vectors of absolute file paths for each image in the dataset. 

This setup is essential for organizing the data and ensuring that our model can access the correct images and labels during training and evaluation.

#### Label Structure

The data set we are using consists of three folders: `train`, `val`, `test`. Each of them contain a set of images of cats. The labels in this case, are in the form of a CSV file that maps the filename with a one-hot encoding to label the classification of the image, i.e. the cat's mood - alarmed, angry, calm, pleased.

The dataset was obtained [here](https://universe.roboflow.com/mubbarryz/domestic-cats-facial-expressions).


```julia
# specify the paths to the dataset and labels CSV
train_data_path = "data/cat_expression_data/train"
train_data_csv = "data/cat_expression_data/train/_classes.csv"

val_data_path = "data/cat_expression_data/val"
val_data_csv = "data/cat_expression_data/val/_classes.csv"

test_data_path = "data/cat_expression_data/test"
test_data_csv = "data/cat_expression_data/test/_classes.csv"

# load the CSV file containing the labels
train_labels_df = CSV.read(train_data_csv, DataFrame)
test_labels_df = CSV.read(test_data_csv, DataFrame)
val_labels_df = CSV.read(val_data_csv, DataFrame)

# setup filepaths to the files as vectors
train_filepaths = [abspath(joinpath(train_data_path, filename)) for filename in train_labels_df[!, 1] ]
test_filepaths = [abspath(joinpath(test_data_path, filename)) for filename in test_labels_df[!, 1] ]
val_filepaths = [abspath(joinpath(val_data_path, filename)) for filename in val_labels_df[!, 1] ]

```




    110-element Vector{String}:
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"18cd56a2ae74d2ffc8fdc89cbb.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"6625698d9d2166cdafe47e6d17.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 106 bytes ⋯ [22m[39m"99a04518d4d80adea474bbe89a.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 104 bytes ⋯ [22m[39m"97c687e09bf5981b9bb729304f.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"be307e32ffc3c27ee7f49305b6.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 106 bytes ⋯ [22m[39m"fabcbee5c45195a0e34918a0a1.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 105 bytes ⋯ [22m[39m"d2b7179bdf5554ea40998d9d93.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 105 bytes ⋯ [22m[39m"bae261f0ca148e055d0935580e.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 102 bytes ⋯ [22m[39m"a84e5fa1564b409f26ea9ed0c9.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 106 bytes ⋯ [22m[39m"4664e5d811b55a69cac9823a87.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"7fccb36f778a5cae5eda1e6cfc.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 104 bytes ⋯ [22m[39m"824395bcb65dc5b8ecd013ab0d.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 106 bytes ⋯ [22m[39m"4e1297350b8f05b54f387e002a.jpg"
     ⋮
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 105 bytes ⋯ [22m[39m"59f6a427983efd9308ddddeea7.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"8768c7f0096dc16431b41c8367.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"aa5d35bce083f1505a7b1e727e.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 104 bytes ⋯ [22m[39m"5c4e68b8fba7c493f0b8bfd7bc.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"05ac086aa8b99cb8b942b1af16.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 102 bytes ⋯ [22m[39m"db819e63e4b80c5caed5a07c47.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"b96481186b7376b108e9546306.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 105 bytes ⋯ [22m[39m"d5aaf15e5d105aa82e24a85eff.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"5e29bd734c42a6b7b2267fb31e.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 104 bytes ⋯ [22m[39m"0628948e8f68a77c821746f0b3.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 103 bytes ⋯ [22m[39m"4dee18ead713b297b872642c25.jpg"
     "H:\\My Drive\\Projects\\Coding\\Por"[93m[1m ⋯ 104 bytes ⋯ [22m[39m"eeffac7d65a76096d457bd5949.jpg"



### Data Exploration

As usual, we take a look at the data to understand what we are working with. 

Below we make a couple of functions to visualize the data.

Note that the helper function `label_from_row` will come in handy later on.


```julia

# -----------------------------------------------------------------------#
# helper function to extract label from the DataFrame
function label_from_row(filename, labels_df, label_dict)
    # retrieve the label for the image from the DataFrame
    label_row = filter(row -> row.filename == filename, labels_df)
    label_index = findfirst(x -> label_row[1, x] == 1, names(labels_df)[2:end])
    
    return label_dict[label_index]
end
# -----------------------------------------------------------------------#


# function to display a selection of images and their labels
function show_sample_images_and_labels(labels_df, label_dict; num_samples = 4)
    # randomly pick indices for sampling images
    sample_indices = rand(1:nrow(labels_df), num_samples)
    sample_filenames = labels_df.filename[sample_indices]

    # calculate number of rows and columns for the grid layuot
    num_cols = ceil(Int, num_samples / 2)
    num_rows = 2
    
    # prepare a plot with a grid layout for the images
    p = plot(layout = (num_rows, num_cols), size(800, 200), legend = false, axis = false, grid = false)

    # load and plot each sampled image
    for (index, filename) in enumerate(sample_filenames)
        img_path = joinpath(train_data_path, filename)
        img = load(img_path)   # load the image from the file

        # retrieve the label for the image from the DataFrame
        label = label_from_row(filename, labels_df, label_dict)

        plot!(p[index], img, title = label, axis = false)
    end

    display(p)   # display the plot
end



# define a dictionary for label descriptions:
label_dict = Dict(1 => "alarmed", 2 => "angry", 3 => "calm", 4 => "pleased")

# run the function to show images
show_sample_images_and_labels(train_labels_df, label_dict)
```

![svg](/images/20240521_julia_transfer_learning_v5/output_6_0.svg)


### Working with Custom Datasets

When working with custom datasets in Julia, the concepts are similar as in PyTorch, but obviously following Julia's syntax. 

In essence, we read the CSV files containing image file paths and their corresponding labels into DataFrames. We then create functions to handle data loading and transformations, such as resizing and normalizing images. This approach is similar to PyTorch's `Dataset`.

Let's have a quick look.

### Create a Custom Dataset

We define a custom dataset using a `struct`, which is similar to using a `class` in Python. The `ImageContainer` struct stores the image file paths and their corresponding labels in a DataFrame. We then create instances of this `struct` for the training, validation, and test datasets.



```julia
struct ImageContainer{T<:Vector}
    img::T
    labels_df::DataFrame
end

# generate dataset
train_dataset = ImageContainer(train_filepaths, train_labels_df);   
val_dataset = ImageContainer(val_filepaths, val_labels_df);
test_dataset = ImageContainer(test_filepaths, test_labels_df);
```

#### Create the Data Loaders

In this section, we set up data loaders for our custom dataset in Julia, similar to how data loaders are used in PyTorch to manage batching and shuffling of data.

1. Call helper Function: `label_from_row()` : This function extracts the label from the DataFrame for a given image file. It finds the index of the column with a value of 1, indicating the class.

2. Length and Indexing:

- `length(data::ImageContainer)`: Defines the length method to return the number of images in the dataset. Similar to PyTorch's `__len__`.
- `getindex(data::ImageContainer, idx::Int)`: This method is similar to PyTorch’s `__getitem__`. It loads an image, applies transformations, and returns the processed image along with its label.
 
3. Data Augmentation and Transformations:

- pipeline: Defines a transformation pipeline for scaling and cropping images.
- transforms(image, labels_df): Inside getindex, this function applies the transformations to the image and normalizes it using the predefined mean and standard deviation values.

4. DataLoaders:

- `train_loader` and `val_loader`: These DataLoader objects manage batching, shuffling, and parallel processing of the training and validation datasets, similar to `torch.utils.data.DataLoader` in PyTorch

##### Notes on Implementing Custom Data Containers

According to the documentation for MLUtils.DataLoader ([see here](https://fluxml.ai/Flux.jl/stable/data/mlutils/)), custom data containers should implement Base.length instead of  `numobs`, and Base.getindex instead of `getobs`, unless there's a difference between these functions and the base methods for multi-dimensional arrays.

Base.length: Should be implemented to return the number of observations. This is akin to PyTorch's `__len__`.
Base.getindex: Should be implemented to handle indexing of the dataset, similar to PyTorch's `__getitem__`.
These methods ensure that the data is returned in a form suitable for the learning algorithm, maintaining consistency whether the index is a scalar or vector.


```julia
length(data::ImageContainer) = length(data.img)

const im_size = (224, 224)
const DATA_MEAN = [0.485f0, 0.456f0, 0.406f0]
const DATA_STD = [0.229f0, 0.224f0, 0.225f0]

# define a transformation pipeline
pipeline = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))

function getindex(data::ImageContainer, idx::Int)
    image = data.img[idx]
    labels_df = data.labels_df
    
    function transforms(image, labels_df)
        pipeline = ScaleKeepAspect(im_size) |> CenterCrop(im_size)
        _img = Images.load(image)
        _img = apply(pipeline, Image(_img)) |> itemdata
        img = collect(channelview(float32.(RGB.(_img))))
        img = permutedims((img .- DATA_MEAN) ./ DATA_STD, (3, 2, 1) )

        label = label_from_row(labels_df[idx, 1] , labels_df)
        return img, label
    end
    
    return transforms(image, labels_df)
end

train_loader = DataLoader(
    train_dataset;
    batchsize = 16,
    collate = true,
    parallel = true,
    )

val_loader = DataLoader(
    val_dataset;
    batchsize = 16,
    collate = true,
    parallel = true,
    );

```

## Model Definition

Here we will load the model with `Metalhead.jl` and change the classifier "head" of the architecture to suit our classification need.

We will use this to select the classifier head of the model and change it.

For the fine-tuning portion of this exercise will follow the [model zoo documentation](https://github.com/FluxML/model-zoo/tree/master/tutorials%2Ftransfer_learning):

___
![image.png](/images/20240521_julia_transfer_learning_v5/109ebfef-0cea-49b5-98d5-fcd19f0f9596.png)

---

Let's try it out with the `ResNet18` model. 


```julia
# load the pre-trained model
resnet_model = ResNet(18; pretrain = true).layers

# let's look at the model
resnet_model
```




    Chain(
      Chain(
        Chain(
          Conv((7, 7), 3 => 64, pad=3, stride=2, bias=false),  [90m# 9_408 parameters[39m
          BatchNorm(64, relu),              [90m# 128 parameters[39m[90m, plus 128[39m
          MaxPool((3, 3), pad=1, stride=2),
        ),
        Chain(
          Parallel(
            addact(NNlib.relu, ...),
            identity,
            Chain(
              Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
              BatchNorm(64),                [90m# 128 parameters[39m[90m, plus 128[39m
              NNlib.relu,
              Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
              BatchNorm(64),                [90m# 128 parameters[39m[90m, plus 128[39m
            ),
          ),
          Parallel(
            addact(NNlib.relu, ...),
            identity,
            Chain(
              Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
              BatchNorm(64),                [90m# 128 parameters[39m[90m, plus 128[39m
              NNlib.relu,
              Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
              BatchNorm(64),                [90m# 128 parameters[39m[90m, plus 128[39m
            ),
          ),
        ),
        Chain(
          Parallel(
            addact(NNlib.relu, ...),
            Chain(
              Conv((1, 1), 64 => 128, stride=2, bias=false),  [90m# 8_192 parameters[39m
              BatchNorm(128),               [90m# 256 parameters[39m[90m, plus 256[39m
            ),
            Chain(
              Conv((3, 3), 64 => 128, pad=1, stride=2, bias=false),  [90m# 73_728 parameters[39m
              BatchNorm(128),               [90m# 256 parameters[39m[90m, plus 256[39m
              NNlib.relu,
              Conv((3, 3), 128 => 128, pad=1, bias=false),  [90m# 147_456 parameters[39m
              BatchNorm(128),               [90m# 256 parameters[39m[90m, plus 256[39m
            ),
          ),
          Parallel(
            addact(NNlib.relu, ...),
            identity,
            Chain(
              Conv((3, 3), 128 => 128, pad=1, bias=false),  [90m# 147_456 parameters[39m
              BatchNorm(128),               [90m# 256 parameters[39m[90m, plus 256[39m
              NNlib.relu,
              Conv((3, 3), 128 => 128, pad=1, bias=false),  [90m# 147_456 parameters[39m
              BatchNorm(128),               [90m# 256 parameters[39m[90m, plus 256[39m
            ),
          ),
        ),
        Chain(
          Parallel(
            addact(NNlib.relu, ...),
            Chain(
              Conv((1, 1), 128 => 256, stride=2, bias=false),  [90m# 32_768 parameters[39m
              BatchNorm(256),               [90m# 512 parameters[39m[90m, plus 512[39m
            ),
            Chain(
              Conv((3, 3), 128 => 256, pad=1, stride=2, bias=false),  [90m# 294_912 parameters[39m
              BatchNorm(256),               [90m# 512 parameters[39m[90m, plus 512[39m
              NNlib.relu,
              Conv((3, 3), 256 => 256, pad=1, bias=false),  [90m# 589_824 parameters[39m
              BatchNorm(256),               [90m# 512 parameters[39m[90m, plus 512[39m
            ),
          ),
          Parallel(
            addact(NNlib.relu, ...),
            identity,
            Chain(
              Conv((3, 3), 256 => 256, pad=1, bias=false),  [90m# 589_824 parameters[39m
              BatchNorm(256),               [90m# 512 parameters[39m[90m, plus 512[39m
              NNlib.relu,
              Conv((3, 3), 256 => 256, pad=1, bias=false),  [90m# 589_824 parameters[39m
              BatchNorm(256),               [90m# 512 parameters[39m[90m, plus 512[39m
            ),
          ),
        ),
        Chain(
          Parallel(
            addact(NNlib.relu, ...),
            Chain(
              Conv((1, 1), 256 => 512, stride=2, bias=false),  [90m# 131_072 parameters[39m
              BatchNorm(512),               [90m# 1_024 parameters[39m[90m, plus 1_024[39m
            ),
            Chain(
              Conv((3, 3), 256 => 512, pad=1, stride=2, bias=false),  [90m# 1_179_648 parameters[39m
              BatchNorm(512),               [90m# 1_024 parameters[39m[90m, plus 1_024[39m
              NNlib.relu,
              Conv((3, 3), 512 => 512, pad=1, bias=false),  [90m# 2_359_296 parameters[39m
              BatchNorm(512),               [90m# 1_024 parameters[39m[90m, plus 1_024[39m
            ),
          ),
          Parallel(
            addact(NNlib.relu, ...),
            identity,
            Chain(
              Conv((3, 3), 512 => 512, pad=1, bias=false),  [90m# 2_359_296 parameters[39m
              BatchNorm(512),               [90m# 1_024 parameters[39m[90m, plus 1_024[39m
              NNlib.relu,
              Conv((3, 3), 512 => 512, pad=1, bias=false),  [90m# 2_359_296 parameters[39m
              BatchNorm(512),               [90m# 1_024 parameters[39m[90m, plus 1_024[39m
            ),
          ),
        ),
      ),
      Chain(
        AdaptiveMeanPool((1, 1)),
        MLUtils.flatten,
        Dense(512 => 1000),                 [90m# 513_000 parameters[39m
      ),
    ) [90m        # Total: 62 trainable arrays, [39m11_689_512 parameters,
    [90m          # plus 40 non-trainable, 9_600 parameters, summarysize [39m44.654 MiB.



Now we modify the head, by chaning the last `Chain` in the model. We change the last layer to output 4 classes (as opposed to the original 1000 classes).


```julia
# modify the model
resnet_infer = deepcopy(resnet_model[1])
resnet_tune = Chain(AdaptiveMeanPool((1, 1)), Flux.flatten, Dense(512 => 4))
```




    Chain(
      AdaptiveMeanPool((1, 1)),
      Flux.flatten,
      Dense(512 => 4),                      [90m# 2_052 parameters[39m
    ) 



**And that's it!** Now, let's just explore both portions of the model.


```julia
resnet_infer
```




    Chain(
      Chain(
        Conv((7, 7), 3 => 64, pad=3, stride=2, bias=false),  [90m# 9_408 parameters[39m
        BatchNorm(64, relu),                [90m# 128 parameters[39m[90m, plus 128[39m
        MaxPool((3, 3), pad=1, stride=2),
      ),
      Chain(
        Parallel(
          addact(NNlib.relu, ...),
          identity,
          Chain(
            Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
            BatchNorm(64),                  [90m# 128 parameters[39m[90m, plus 128[39m
            NNlib.relu,
            Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
            BatchNorm(64),                  [90m# 128 parameters[39m[90m, plus 128[39m
          ),
        ),
        Parallel(
          addact(NNlib.relu, ...),
          identity,
          Chain(
            Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
            BatchNorm(64),                  [90m# 128 parameters[39m[90m, plus 128[39m
            NNlib.relu,
            Conv((3, 3), 64 => 64, pad=1, bias=false),  [90m# 36_864 parameters[39m
            BatchNorm(64),                  [90m# 128 parameters[39m[90m, plus 128[39m
          ),
        ),
      ),
      Chain(
        Parallel(
          addact(NNlib.relu, ...),
          Chain(
            Conv((1, 1), 64 => 128, stride=2, bias=false),  [90m# 8_192 parameters[39m
            BatchNorm(128),                 [90m# 256 parameters[39m[90m, plus 256[39m
          ),
          Chain(
            Conv((3, 3), 64 => 128, pad=1, stride=2, bias=false),  [90m# 73_728 parameters[39m
            BatchNorm(128),                 [90m# 256 parameters[39m[90m, plus 256[39m
            NNlib.relu,
            Conv((3, 3), 128 => 128, pad=1, bias=false),  [90m# 147_456 parameters[39m
            BatchNorm(128),                 [90m# 256 parameters[39m[90m, plus 256[39m
          ),
        ),
        Parallel(
          addact(NNlib.relu, ...),
          identity,
          Chain(
            Conv((3, 3), 128 => 128, pad=1, bias=false),  [90m# 147_456 parameters[39m
            BatchNorm(128),                 [90m# 256 parameters[39m[90m, plus 256[39m
            NNlib.relu,
            Conv((3, 3), 128 => 128, pad=1, bias=false),  [90m# 147_456 parameters[39m
            BatchNorm(128),                 [90m# 256 parameters[39m[90m, plus 256[39m
          ),
        ),
      ),
      Chain(
        Parallel(
          addact(NNlib.relu, ...),
          Chain(
            Conv((1, 1), 128 => 256, stride=2, bias=false),  [90m# 32_768 parameters[39m
            BatchNorm(256),                 [90m# 512 parameters[39m[90m, plus 512[39m
          ),
          Chain(
            Conv((3, 3), 128 => 256, pad=1, stride=2, bias=false),  [90m# 294_912 parameters[39m
            BatchNorm(256),                 [90m# 512 parameters[39m[90m, plus 512[39m
            NNlib.relu,
            Conv((3, 3), 256 => 256, pad=1, bias=false),  [90m# 589_824 parameters[39m
            BatchNorm(256),                 [90m# 512 parameters[39m[90m, plus 512[39m
          ),
        ),
        Parallel(
          addact(NNlib.relu, ...),
          identity,
          Chain(
            Conv((3, 3), 256 => 256, pad=1, bias=false),  [90m# 589_824 parameters[39m
            BatchNorm(256),                 [90m# 512 parameters[39m[90m, plus 512[39m
            NNlib.relu,
            Conv((3, 3), 256 => 256, pad=1, bias=false),  [90m# 589_824 parameters[39m
            BatchNorm(256),                 [90m# 512 parameters[39m[90m, plus 512[39m
          ),
        ),
      ),
      Chain(
        Parallel(
          addact(NNlib.relu, ...),
          Chain(
            Conv((1, 1), 256 => 512, stride=2, bias=false),  [90m# 131_072 parameters[39m
            BatchNorm(512),                 [90m# 1_024 parameters[39m[90m, plus 1_024[39m
          ),
          Chain(
            Conv((3, 3), 256 => 512, pad=1, stride=2, bias=false),  [90m# 1_179_648 parameters[39m
            BatchNorm(512),                 [90m# 1_024 parameters[39m[90m, plus 1_024[39m
            NNlib.relu,
            Conv((3, 3), 512 => 512, pad=1, bias=false),  [90m# 2_359_296 parameters[39m
            BatchNorm(512),                 [90m# 1_024 parameters[39m[90m, plus 1_024[39m
          ),
        ),
        Parallel(
          addact(NNlib.relu, ...),
          identity,
          Chain(
            Conv((3, 3), 512 => 512, pad=1, bias=false),  [90m# 2_359_296 parameters[39m
            BatchNorm(512),                 [90m# 1_024 parameters[39m[90m, plus 1_024[39m
            NNlib.relu,
            Conv((3, 3), 512 => 512, pad=1, bias=false),  [90m# 2_359_296 parameters[39m
            BatchNorm(512),                 [90m# 1_024 parameters[39m[90m, plus 1_024[39m
          ),
        ),
      ),
    ) [90m        # Total: 60 trainable arrays, [39m11_176_512 parameters,
    [90m          # plus 40 non-trainable, 9_600 parameters, summarysize [39m42.693 MiB.




```julia
resnet_tune
```




    Chain(
      AdaptiveMeanPool((1, 1)),
      Flux.flatten,
      Dense(512 => 4),                      [90m# 2_052 parameters[39m
    ) 



### Define evaluation and training functions

Again, will follow the model zoo documentation. Small adaptations will be needed. (These two functions were taken directly from the documentation).


```julia
function eval_f(m_infer, m_tune, val_loader)
    good = 0
    count = 0
    for(x, y) in val_loader
        good += sum(Flux.onecold(m_tune(m_infer(x))) .== y)
        count += length(y)
    end
    acc = round(good / count, digits = 4)
    return acc
end
```




    eval_f (generic function with 1 method)




```julia
function train_epoch!(model_infer, model_tune, opt, loader)
    for (x, y) in loader
        infer = model_infer(x)
        grads = gradient(model_tune) do m
            Flux.Losses.logitcrossentropy(m(infer), Flux.onehotbatch(y, 1:4))
        end
        update!(opt, model_tune, grads[1])
    end
end

```




    train_epoch! (generic function with 1 method)




```julia
resnet_opt = Flux.setup(Flux.Optimisers.Adam(1e-3), resnet_tune);
```


```julia
for iter = 1:5
    @time train_epoch!(resnet_infer, resnet_tune, resnet_opt, train_loader)
    metric_train = eval_f(resnet_infer, resnet_tune, train_loader)
    metric_eval = eval_f(resnet_infer, resnet_tune, val_loader)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end
```

    176.283332 seconds (37.11 M allocations: 98.153 GiB, 6.06% gc time, 143.87% compilation time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.5744
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.5455
    

     70.815518 seconds (2.42 M allocations: 95.936 GiB, 11.25% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.6823
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6273
    

     90.463025 seconds (2.42 M allocations: 95.936 GiB, 11.21% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.7032
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6455
    

     94.362892 seconds (2.42 M allocations: 95.936 GiB, 10.91% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.7433
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6727
    

    116.526515 seconds (2.42 M allocations: 95.936 GiB, 9.62% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.7885
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6909
    

---
## Vision Transformers
---

Similar to the PyTorch demonstration, we can do transfer learning by changing a different computer vision model (Vision Transformer).

Let's get into it.


```julia
vit_model = ViT(:base; pretrain = true).layers

# let's have a look at the model head, to see how many inputs the head needs
vit_model[2]
```




    Chain(
      LayerNorm(768),                       [90m# 1_536 parameters[39m
      Dense(768 => 1000),                   [90m# 769_000 parameters[39m
    ) [90m                  # Total: 4 arrays, [39m770_536 parameters, 2.940 MiB.




```julia
# modify the head
vit_infer = deepcopy(vit_model[1])

# notice how we keep the input to the model head
vit_tune = Chain(
    LayerNorm(768),
    Dense(768 => 4),
    )
```




    Chain(
      LayerNorm(768),                       [90m# 1_536 parameters[39m
      Dense(768 => 4),                      [90m# 3_076 parameters[39m
    ) [90m                  # Total: 4 arrays, [39m4_612 parameters, 18.352 KiB.




```julia
vit_opt = Flux.setup(Flux.Optimisers.Adam(1e-3), vit_tune);
```


```julia
for iter = 1:5
    @time train_epoch!(vit_infer, vit_tune, vit_opt, train_loader)
    metric_train = eval_f(vit_infer, vit_tune, train_loader)
    metric_eval = eval_f(vit_infer, vit_tune, val_loader)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end
```

    627.303072 seconds (17.32 M allocations: 291.924 GiB, 4.61% gc time, 3.66% compilation time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.7058
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6273
    

    565.986959 seconds (2.54 M allocations: 291.028 GiB, 4.71% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.8042
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6273
    

    516.041945 seconds (2.54 M allocations: 291.028 GiB, 4.92% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.866
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6818
    

    515.415614 seconds (2.54 M allocations: 291.028 GiB, 4.80% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.8973
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6818
    

    427.423410 seconds (2.54 M allocations: 291.028 GiB, 5.01% gc time)
    

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mtrain
    [36m[1m└ [22m[39m  metric = 0.9199
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39meval
    [36m[1m└ [22m[39m  metric = 0.6727
    

### Save the Models


```julia
using JLD2

resnet_model_state = Flux.state(resnet_model)
vit_model_state = Flux.state(vit_model)

jldsave("resnet_model.jld2"; resnet_model_state)
jldsave("vit_model.jld2"; vit_model_state)
```

    [33m[1m┌ [22m[39m[33m[1mWarning: [22m[39mOpening file with JLD2.MmapIO failed, falling back to IOStream
    [33m[1m└ [22m[39m[90m@ JLD2 C:\Users\ingvi\.julia\packages\JLD2\7uAqU\src\JLD2.jl:300[39m
    [33m[1m┌ [22m[39m[33m[1mWarning: [22m[39mOpening file with JLD2.MmapIO failed, falling back to IOStream
    [33m[1m└ [22m[39m[90m@ JLD2 C:\Users\ingvi\.julia\packages\JLD2\7uAqU\src\JLD2.jl:300[39m
    


```julia
using BSON: @save

@save "resnet_model_sate.bson" resnet_model
@save "vit_model_state.bson" vit_model
```

## Thank you!

I hope this demonstration on using Julia and `Flux` for transfer learning was helpful!

Victor
