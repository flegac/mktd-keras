# Project Keras

Train a model to recognize monkeys !
- https://www.kaggle.com/slothkong/10-monkey-species

## Objectif :

0. Visualize / inspect your dataset
1. Create a basic Model & train it against the dataset
2. Package the Model into a web service to make predictions
3. Improve your model / test it against images found on-line !

## Notes:
Multiple models can be trained by different members of the team.
The prediction web service can improve its performances just by averaging the predictions of multiples models.


## Instructions

0. Create a [script](project/visualize.py) to visualize 5 images for each category.
1. Use [transfer learning](https://keras.io/applications/) to fit an existing model on your dataset.
   Use the [training script](./project/training/training_script.py)
2. Load your model in a [web service](./project/server.py) using Flask (or whatever you like, must be compatible with keras models)
3. Use the [test script](./tests/test_server.py) to test your model against random images.
