# Solo or Ensemble? Choosing a CNN Architecture for Melanoma Classification

This project contains the source code of all experiments described in
'Solo or Ensemble? Choosing a CNN Architecture for Melanoma Classification.'


## Abstract

> Convolutional neural networks (CNNs) deliver exceptional
> results for computer vision, including medical image analysis.
> With the growing number of available architectures, picking
> one over another is far from obvious. Existing art suggests
> that, when performing transfer learning, the performance of
> CNN architectures on ImageNet correlates strongly with their
> performance on target tasks.
> We evaluate that claim for melanoma classification, over 9
> CNNs architectures, in 5 sets of splits created on the ISIC
> Challenge 2017 dataset, and 3 repeated measures, resulting in
> 135 models. The correlations we found were, to begin with,
> much smaller than those reported by existing art, and
> disappeared altogether when we considered only the
> top-performing networks: uncontrolled nuisances (i.e., splits
> and randomness) overcome any of the analyzed factors. Whenever
> possible, the best approach for melanoma classification is
> still to create ensembles of multiple models. We compared two
> choices for selecting which models to ensemble: picking them
> at random (among a pool of highquality ones) vs. using the
> validation set to determine which ones to pick first.
> For small ensembles, we found a slight advantage on the second
> approach but found that random choice was also competitive.
> Although our aim in this paper was not to maximize performance,
> we easily reached AUCs comparable to the first place on the
> ISIC Challenge 2017.


## Project setup

1. Install OpenCV with `pip3 install opencv-python`.
2. Run `pip3 install -r requirements.txt`.


### Telegram API

The project uses [Sacred](http://sacred.readthedocs.io) to organize the
experiments. If you want to monitor the experiments with Telegram (receive a message when
the experiments start, finish, or fail), create a file `telegram.json` at the
root of the project:

```
$ cat telegram.json
{
    "token": "00000000:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    "chat_id": "00000000"
}
```

To configure the Telegram API, check
[this](https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id).


## Reproducing the paper

Register to [ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a).

Download data for the three phases of 'Part 3: Lesion Classification' ([training](https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab), [validation](https://challenge.kitware.com/#phase/584b0afacad3a51cc66c8e33), [testing](https://challenge.kitware.com/#phase/584b0afccad3a51cc66c8e38).

To make train faster, we resize every image to a maximum width or height of
1024 pixels. This will make augmentation operations (e.g. resizing, random
crop) run faster.
[ImageMagick](http://imagemagick.org/script/index.php) can do the trick:

```
cd ISIC-2017_Training_Data
mkdir 1024
convert "*.jpg[1024x>]" -set filename:base "%[base]" "1024/%[filename:base].jpg"
```

Repeat the same for the validation and test sets.

Put all images (train, validation, test) into the same directory
(e.g. `ISIC2017_Full`). We will use our own splits.

Set the path to `ISIC2017_Full`:

```
export ISIC2017_FULL=path/to/ISIC2017_Full
```

Now, run all experiments:

```
./experiments/run.sh
```

## Ensembles

```
python3 create_ensembles.py
```

This will create three types of ensembles:

1) Average models with same architecture and same split.
2) Average models with different architectures and same split, at random.
3) Average models with different architectures and same split, sorted by
   validation AUC.

The files will be available at `results/ensemble_{1,2,3}.csv`.


## Statistics

The code will be available in the next few weeks.


## Cite

If you use this code, please cite us:

```
@inproceedings{perez19soloorensemble,
 author    = {Perez, F{\'a}bio and and Avila, Sandra and Valle, Eduardo},
 title     = {Solo or Ensemble? Choosing a CNN Architecture for Melanoma Classification},
 booktitle = {ISIC Skin Image Anaylsis Workshop, 2019 {IEEE} Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
 year      = {2019},
}
```
