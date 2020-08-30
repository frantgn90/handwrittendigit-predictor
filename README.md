# Handwritten digits predictor
This is a naive and suboptimal implementation of a handwritten digits prediction. 
Under the hood there is a non-optimized single-layer perceptron that uses the delta rule for learning

## Train the model
In order to train the model you shoud run `./perceptron.py --train`. It is gonna train the model using the MNIST
database. It would expect the files to be placed under `data/`. The paths are hardcoded so if you want to use
a different ones you should go to the code. When the training is done, the weights will be stored in 
`weights/<current-timestamp>.wgt`. The `weights` folder must exists.

## Run the GUI
You can run the GUI and start perform predictions on handwritten digits by executing `./perceptron --weights <weights-path>`
Three considerations about the GUI:
- To write a digit click the mouse's left button and drag
- To perform a prediction on the written digit double-click anywhere in the canvas
- To load the next test handwritten digit click the mouse's right button
- There is an extra pre-process that is in TODO yet (`_rescale_to_20x20`). 
Because of that if you want to have a better performance on the prediction, please write the digit in the white frame in the canvas.

## Dependencies
- numpy
- scipy
- tkinter
