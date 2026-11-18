# Maschinelles Lernen – Traffic Sign Classifier

A C# console application that classifies images of traffic signs. It extracts a feature vector from every image,
trains one of five learning methods on those vectors, saves the trained model and then classifies new images.
It can also train a method many times to measure how stable its accuracy is, sweep a complexity setting
(like the number of hidden layers or k) to see how it changes the accuracy, or test hand-edited network weights
on the same splits.

## Contents

- [Requirements](#requirements)
- [Project layout](#project-layout)
- [Training data](#training-data)
- [Running the application](#running-the-application)
- [Learning methods](#learning-methods)
- [Options](#options)
- [Examples](#examples)
- [Output files](#output-files)
- [Editing the weights of a network](#editing-the-weights-of-a-network)
- [How a run works](#how-a-run-works)
- [Default settings](#default-settings)
- [Troubleshooting](#troubleshooting)

## Requirements

- .NET 10 SDK (the project targets `net10.0`)
- The SkiaSharp NuGet package is restored automatically on the first build
- Training images, see [Training data](#training-data)

## Project layout

```
uni_machine_learning_cs.sln
Agents.md                          instructions for coding agents working on the repository
results.numbers                    charts of earlier large runs
uni_machine_learning_cs/           the project
  Program.cs                       command line handling, training and classification
  Colors.cs                        the traffic sign colours used by the colour presence features
  ImageAnalysis/ReadImageFile.cs   reads BMP / JPEG images
  MachineLearning/
    FeatureExtractor.cs            turns an image into its feature vector
    FeatureFile.cs                 the format of image_data.json
    ModelTrainer.cs                splits the data, trains and evaluates a method, runs and sweeps
    TrainedModel.cs                a trained classifier that can be saved, loaded and used
    NeuralNetwork.cs, KNearestNeighbors.cs, DecisionTree.cs,
    LogicGateNetwork.cs, Perceptron.cs
    IClassifier.cs                 the interface every learning method implements
    NetworkWeightsFile.cs          the editable weights JSON of a network / perceptron
    BinaryExtensions.cs            helpers to write / read arrays in model files
  RightTurn/, StopSign/, ...       the training images, one folder per concept
image_data.json                    the extracted feature vectors (generated)
models/                            trained models (generated)
evaluations/                       results of --runs and --sweep as CSV (generated)
```

The generated files are written next to the `.sln` file, no matter where the program is started from.
Images, models, evaluations and `image_data.json` are ignored by git.

## Training data

Each concept has one folder inside `uni_machine_learning_cs/`. All images in that folder, including subfolders,
are used as examples of that concept. Supported extensions are `.bmp`, `.jpg` and `.jpeg`.

| Folder                  | Concept       |
|-------------------------|---------------|
| `RightTurn`             | right turn    |
| `StopSign`              | stop sign     |
| `Vorfahrt von rechts`   | right of way  |
| `Fahrtrichtung links`   | left turn     |
| `Vorfahrt gewähren`     | yield         |

The folders are listed in `conceptFolders` at the top of `Program.cs`. To add a concept, add its folder there and
train again with `--train`.

BMPs must be uncompressed 24 or 32 bit. JPEGs and other sizes are decoded and scaled to the size of the BMPs.

## Running the application

Run from inside the project folder, because the concept folders are looked up relative to the current directory.
Options for the application go after `--`, otherwise `dotnet` reads them itself.

```bash
cd uni_machine_learning_cs
dotnet run -- <options> [<image> ...]
```

Without any options the program asks which learning method to use, trains it if no model is saved yet, and then
asks for images to classify one at a time. Press Enter on an empty line to quit.

There are these modes:

| Mode                             | What it does                                                                                       |
|----------------------------------|----------------------------------------------------------------------------------------------------|
| `<method> [<image> ...]`         | Loads the saved model of the method (trains it first if there is none) and classifies images       |
| `--train <method> [<image> ...]` | Extracts the features of all images again, trains the method, saves the model, then classifies     |
| `--runs <n> <method>`            | Trains and tests the method n times with different random splits, saves the accuracies to a CSV    |
| `--runs <n> <method> --weights`  | Trains nothing: tests the saved model, with its edited weights, on the test data of the same n splits |
| `--sweep <n> <method>`           | Like `--runs`, but repeated for several complexity values of the method, saves one CSV             |

## Learning methods

Exactly one method is chosen with one of these options. If none is given, the program asks.

| Option         | Method                                  | Complexity setting used by `--sweep`  | Model file               |
|----------------|-----------------------------------------|---------------------------------------|--------------------------|
| `--nn`         | Neural network (multi-layer perceptron) | number of hidden layers               | `neural_network`         |
| `--knn`        | k-nearest neighbors                     | k, the number of voting neighbors     | `k_nearest_neighbors`    |
| `--tree`       | Decision tree                           | maximum depth                         | `decision_tree`          |
| `--gates`      | Logic gate network                      | number of gate layers                 | `logic_gate_network`     |
| `--perceptron` | Single-layer perceptron                 | number of training epochs             | `perceptron_<variant>`   |

**Logic gate network:** every neuron learns to be one of the 16 two-input logic gates. After training, the
finished circuit is written to a text file (see [Output files](#output-files)).

**Perceptron variants:** the perceptron is trained with the perceptron learning rule and has two switches that
can be combined. Every combination is saved as its own model.

| Option     | Effect                                                                                                      |
|------------|-------------------------------------------------------------------------------------------------------------|
| (default)  | Multi-class rule: only the correct output neuron and the wrongly winning neuron learn from a mistake        |
| `--ovr`    | One-vs-rest: the classic rule, every output neuron learns "my concept or not" on its own                    |
| `--pocket` | Keeps the weights with the best training accuracy seen after any epoch instead of the last ones             |

The model name is `perceptron_multiclass`, `perceptron_one_vs_rest`, `perceptron_multiclass_pocket` or
`perceptron_one_vs_rest_pocket`.

## Options

| Option                   | Meaning                                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------------------|
| `--train`                | Re-extract the features of all images and train the chosen method, even if a model is already saved        |
| `--runs <n>`             | Train and test n times with random splits (seeds 1..n), write the accuracies to `evaluations/`              |
| `--sweep <n>`            | Run n times for every complexity value, write the accuracies per value to `evaluations/`                   |
| `--values 1,2,3`         | The complexity values to try with `--sweep` (comma separated, no spaces). Defaults are used without it     |
| `--weights`              | With `--runs`: test the saved `--nn` / `--perceptron` model with its `_weights.json` instead of training |
| `--binary`               | Turn every feature into yes/no before training: 1 if above its average in the training images, else 0      |
| `--ovr`, `--pocket`      | Perceptron variants, see above                                                                             |
| `<image> ...`            | Paths of images to classify. Without any, the program asks for them interactively                          |

`--binary` works with every method and is saved as its own model, e.g. `neural_network_binary.model`. For the gate
network it yields a pure logic circuit; without it the gates work on 0-1 values like probabilities.

Unknown options starting with `--` stop the program with an error.

## Examples

```bash
cd uni_machine_learning_cs

# Train the neural network and classify two images with it
dotnet run -- --train --nn ../some/stop.bmp ../some/turn.jpg

# Use the saved k-NN model (trained on the first run) and classify interactively
dotnet run -- --knn

# Train a decision tree on yes/no features
dotnet run -- --train --tree --binary

# Perceptron with the one-vs-rest rule and the pocket algorithm
dotnet run -- --train --perceptron --ovr --pocket

# 100 runs of k-NN with different random splits, results go to evaluations/
dotnet run -- --runs 100 --knn

# Test the saved perceptron with hand-edited weights on the same 100 splits
dotnet run -- --runs 100 --perceptron --weights

# Sweep the number of hidden layers of the neural network, 5 runs per value
dotnet run -- --sweep 5 --nn

# Sweep k with your own values
dotnet run -- --sweep 10 --knn --values 1,3,5,11,21

# Sweep the number of epochs of the perceptron with yes/no features
dotnet run -- --sweep 5 --perceptron --binary --values 1,10,100
```

Classifying an image prints the detected concept and the probability of every concept:

```
../some/stop.bmp: stop sign
  right turn   2.1 %
  stop sign    95.3 %
  right of way 0.8 %
  left turn    0.6 %
  yield        1.2 %
```

## Output files

All generated files are written to the folder that holds the `.sln` file.

| File                                                  | Written by                       | Contents                                                                 |
|-------------------------------------------------------|----------------------------------|--------------------------------------------------------------------------|
| `image_data.json`                                     | `--train`, first run of any mode | Author, feature names and one feature vector per training image         |
| `models/<model name>.model`                           | training                         | The trained classifier plus the concept names and feature scaling       |
| `models/<model name>_weights.json`                    | training `--nn` / `--perceptron` | The bias and weights of every neuron, editable (see below)              |
| `models/<model name>_circuit.txt`                     | training `--gates`               | Every gate of the finished circuit, layer by layer                      |
| `evaluations/<model name>_<n>_runs_<date>.csv`        | `--runs`                         | One line per run: `run,seed,test accuracy`                              |
| `evaluations/<model name>_weights_<n>_runs_<date>.csv`| `--runs --weights`               | Same format, accuracies of the saved model with its edited weights      |
| `evaluations/<model name>_sweep_<n>_runs_<date>.csv`  | `--sweep`                        | One line per value: value, mean, standard deviation, min, max, every run |

`image_data.json` is only extracted once; later runs reuse it. Use `--train` to extract it again, for example
after adding images. `--runs` and `--sweep` never overwrite a saved model.

The CSV files use `.` as the decimal separator and can be opened in a spreadsheet directly. The `results.numbers`
file in the project root holds the charts of earlier large runs.

Example of a sweep CSV:

```
epochs,mean,standard deviation,min,max,run 1,run 2,run 3,run 4,run 5
1,0.6922,0.0569,0.6082,0.7599,0.6767,0.7516,0.6082,0.7599,0.6645
5,0.7130,0.0467,0.6698,0.7993,0.7993,0.7235,0.6903,0.6821,0.6698
```

Example of a circuit file (with `--binary`):

```
# Layer 1
L1G0 = [cell 3,3 average r above average] XOR [vertical line 10 switches above average]
L1G1 = [cell 5,4 average b above average]
L1G2 = TRUE
L1G4 = [cell 0,4 average g above average] OR NOT [cell 2,1 average b above average]
```

## Editing the weights of a network

Training the neural network or the perceptron writes `models/<model name>_weights.json`. Every layer lists its
neurons by name with a bias and one weight per input. For the perceptron the inputs are the features themselves
(`"red"`, `"cell 0,0 average r"`, ...). For the neural network the output neurons take their inputs from the
hidden neurons (`"hidden 1 neuron 0"`, ...).

```json
{
  "layers": [
    {
      "name": "output",
      "neurons": [
        {
          "name": "stop sign",
          "bias": 30,
          "weights": {
            "red": 12.5,
            "vertical line 4 switches": 165.7
          }
        }
      ]
    }
  ]
}
```

Every time the saved model is loaded, the weights in this file are applied on top of it. Only what is listed is
changed, so neurons and weights that are left out keep their trained value. The file is overwritten the next time
the method is trained with `--train`. A file that does not match the model, for example a weight for an unknown
input, stops the program with an error.

### Testing edited weights

To see how good the edited weights are, run `--runs <n>` with `--weights` and the same method options as the
model, for example `dotnet run -- --runs 100 --perceptron --pocket --weights`. Nothing is trained: the saved model
is loaded, the weights file is applied, and the model is tested on the test data of the splits with seeds 1..n,
the same splits that `--runs <n>` without `--weights` uses. That way the accuracies can be compared run by run with
freshly trained models. The accuracy of every run and a confusion matrix summed over all runs are printed and saved
to `evaluations/<model name>_weights_<n>_runs_<date>.csv`.

This only works with the neural network and the perceptron, and needs a saved model (train one with `--train`
first). Keep in mind that the saved model was trained on its own split (seed 50), so some of the test images of
these runs were part of its training data, which makes the accuracies a bit too optimistic.

## How a run works

1. **Feature extraction.** Every image is turned into a vector of numbers on all CPU cores. The features are the
   image size, the number of colour switches along 11 vertical and 11 horizontal lines, whether each of eight
   colours is present, the average colour of the image and the average colour of each cell in an 8 x 8 grid.
   The names of all features are listed in `FeatureExtractor.Properties`.
2. **Split.** The vectors are shuffled with a seed and split into training and test data (70 % / 30 %). With `--runs` and
   `--sweep` the seed is the run number, so every complexity value is tested on the same splits.
3. **Scaling.** Every feature is scaled to 0-1 with the minimum and maximum of the training data. With `--binary`
   it is then turned into 1 if it is above its training average, otherwise 0.
4. **Training and evaluation.** The method is trained on the training data. The progress, the test accuracy and a
   confusion matrix are printed. The model remembers the scaling so new images are prepared the same way.
5. **Classification.** A new image goes through the same feature extraction and scaling and is classified. The
   result is the winning concept plus a probability for every concept.

A saved model that can no longer be read, for example because the feature set changed since it was trained, is
trained again automatically.

## Default settings

All of these are constants at the top of `MachineLearning/ModelTrainer.cs`.

| Setting                                     | Value                                   |
|---------------------------------------------|-----------------------------------------|
| Share of the images used for training       | 70 % (30 % for testing)                 |
| Seed of a single training run               | 50                                      |
| Neural network                              | 1 hidden layer, 32 neurons, 200 epochs, learning rate 0.01 |
| k-nearest neighbors                         | k = 5                                   |
| Decision tree                               | max depth 20, min 5 samples to split    |
| Logic gate network                          | 4 layers of 1000 gates, 50 epochs, batch 100, learning rate 0.01 |
| Perceptron                                  | at most 100 epochs, stops early once an epoch has no mistakes |

Default values tried by `--sweep` when `--values` is not given:

| Method         | Values                                   |
|----------------|------------------------------------------|
| `--nn`         | 0, 1, 2, 3, 4, 5 hidden layers           |
| `--knn`        | k = 1, 3, 5, 7, 9, 15, 21, 31, 51, 101   |
| `--tree`       | depth 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30 |
| `--gates`      | 1, 2, 3, 4, 6, 8 gate layers             |
| `--perceptron` | 1, 2, 5, 10, 20, 50, 100, 200 epochs     |

## Troubleshooting

- **No images are found / "Wrote 0 vectors".** Start the program from inside `uni_machine_learning_cs/`, the
  concept folders are resolved relative to the current directory. When running from Rider, set the working
  directory of the run configuration to the project folder.
- **"Unknown option".** Check the spelling, every option starts with `--`. Remember the `--` after `dotnet run`.
- **"--runs needs the number of runs".** The number must directly follow `--runs` or `--sweep`.
- **"--weights needs a saved model" / "--weights only works with ...".** Train the model with `--train` first, and
  use `--weights` only with `--nn` or `--perceptron`.
- **"Could not apply the neuron weights".** The weights JSON does not match the model. Delete it or train again
  with `--train` to get a fresh file.
- **"Could not classify ...".** The image path is wrong, or the BMP is compressed or not 24/32 bit.
- **Old model after changing the features.** Run with `--train`; incompatible saved models are also retrained
  automatically.
