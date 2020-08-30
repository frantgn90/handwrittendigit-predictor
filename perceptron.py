#!/usr/bin/env python3.6

import random
import sys
import time
import json
import os
import argparse
import numpy
import scipy.ndimage

from tkinter import *


class UbtyeDataset(object):
    """ 
    This class provides the proper methods to read the MNIST database ubytes files. 
    It is implemented as an iterator so that the images and labels can be easily accesible
    """
    def __init__(self, images_file_name: str, labels_file_name: str):
        """
        Initialize the class and read the metadata of the ubyte files
        :param images_file_name: ubyte file with images
        :param labels_file_name: ubyte file with the labels for the images
        """
        self.images_file = open(images_file_name, "rb")
        self.labels_file = open(labels_file_name, "rb")
        assert int.from_bytes(self.labels_file.read(4), byteorder="big") == 2049  # It is actually labels file
        assert int.from_bytes(self.images_file.read(4), byteorder="big") == 2051  # It is actually an images file
        self.numberimages = int.from_bytes(self.images_file.read(4), byteorder="big")
        assert int.from_bytes(self.labels_file.read(4), byteorder="big") == self.numberimages  # Both must have the same number of images information
        self.image_rows = int.from_bytes(self.images_file.read(4), byteorder="big")
        self.image_cols = int.from_bytes(self.images_file.read(4), byteorder="big")

        self._current_image = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_image == self.numberimages:
            raise StopIteration
        image = self.images_file.read(self.image_rows * self.image_cols)
        label = int.from_bytes(self.labels_file.read(1), byteorder="big")
        self._current_image += 1
        return image, label

    def __del__(self):
        self.images_file.close()
        self.labels_file.close()


class Perceptron(object):
    """ The perceptron implementation """

    def __init__(self, retina_nn: int, output_nn: int, labels: str, weights_file_path: str = None):
        """ Initialize the perceptron.
        :param retina_nn: The number of neurons in the retina layer.
        :param output_nn: The number of neurons in the output layer.
        :param labels: The labels for the output. Only one output neuron is activated. At that point it would have the
         semantic information specified by the label in the same position.
        :param weights_file_path: Load the weights from here. If it is not specified, random weights will be loaded
        """
        self.learning_constant = 0.1
        self.retina_nn = retina_nn + 1
        self.output_nn = output_nn
        self.labels = labels

        # The bias is added as the first weight for every output neuron. This is why we need retina_nn + 1
        # The bias weight is always activated
        if not weights_file_path:
            self.weights = [[ random.random() for _ in range(self.retina_nn) ] for _ in range(self.output_nn) ]
        else:
            with open(weights_file_path, "r") as wf:
                self.weights = json.load(wf)
            assert len(self.weights) == self.output_nn
            assert len(self.weights[0]) == self.retina_nn
    
    def save_weights(self, weights_file_path: str):
        """ The weights are saved in the path specified in `weights_file_path` as a regular matrix
        :param weights_file_path: The path where to store the weights
        """
        with open(weights_file_path, "w") as wf:
            json.dump(self.weights, wf)

    def teach(self, stimuli: list, expected_label: str) -> bool:
        """ It adjust the weights to fit with the expected result. It uses the perceptron delta rule
        :param stiumuli: The input stimuli that is gonna go to the retina of the perceptron
        :param expected_label: The expected result. It needs to be one of the specified labels in the constructor
        :return: 
        """
        # Adding the bias activation
        stimuli = [1] + stimuli
        output_layer = self._calculate_output_layer(stimuli)
        expected_output_layer = [ 0 ] * len(self.labels)
        expected_output_layer[self.labels.index(expected_label)] = 1

        # Only adjusting the implied outputs. Decrease weights on wrongly activated neuron and increase
        # weights on wrongly deactivated neuron
        try:
            oa = output_layer.index(1)
        except ValueError:
            oa = None
        eoa = expected_output_layer.index(1)
        if oa != eoa:
            for j in range(self.retina_nn):
                if oa is not None:
                    self.weights[oa][j] += self.learning_constant * (expected_output_layer[oa] - output_layer[oa]) * stimuli[j]
                self.weights[eoa][j] += self.learning_constant * (expected_output_layer[eoa] - output_layer[eoa]) * stimuli[j]

    def predict(self, stimuli: list) -> str:
        """ Performs a prediction on an stimuli
        :param stimuli: The simuli with the same size specified in the constructor for the retina size. Is a list of 1 and 0
        :returns: The resut of the prediction according with the labels specified in the constructor
        """
        # Adding the bias activation
        stimuli = [1] + stimuli
        result = self._calculate_output_layer(stimuli)
        try:
            return self.labels[result.index(1)]
        except ValueError:
            return None

    def _calculate_output_layer(self, stimuli: list) -> list:
        """ Performs the calculations for the output layer
        :param stimuli: The data for the retina layer
        :return: The values for the output layer neurons
        """
        result = [ sum([stimuli[i] * self.weights[j][i] for i in range(self.retina_nn)]) for j in range(self.output_nn) ]
        max_value = max(result)

        # Applies a ReLu and a Normalize output
        return [ max(0, output)/max_value for output in result ]

    def train(self, train_data_file: str, train_labels_file: str, test_data_file: str, 
              test_labels_file: str, max_training_epochs: int=2, save_weights: str=True):
        """ Trains the neuron with the data specified in the ubyte files. After every epoch a test agains the test data
        is performed. The operation is repeated until no errors are detected or until `max_training_epochs` is reached.
        :param train_data_file: Path to the train data file
        :param train_labels_file: Path to the train labels file
        :param test_labels_file: Path to the test labels file
        :param max_training_epochs: Max number of epochs to converge
        :param save_weights: Whether to save the weights or not. If true it is gonna be saved in `weights/{current_timestamp}.wgt`
        """
        for trial in range(max_training_epochs):
            print(f"Running training epoch {trial}...")
            for data, tag in UbtyeDataset(train_data_file, train_labels_file):
                self.teach([ 1 if data[i] > 0 else 0 for i in range(self.retina_nn) ], tag)

            fails = 0
            print("Testing against testing data...", end="", flush=True)
            for data, tag in UbtyeDataset(test_data_file, test_labels_file):
                prediction = self.predict([ 1 if data[i] > 0 else 0 for i in range(self.retina_nn) ])
                if not prediction == tag:
                    fails += 1
            print(f" Accuracy: {round((60000-fails)/600, 3)}%")
            if fails == 0:
                break
        if save_weights:
            weights_file = f"weights/{int(time.time())}.wgt"
            perceptron.save_weights(weights_file)
            print(f"Weights saved in: {weights_file}")


class HandWrittenDigitPredictor(Perceptron):
    """
    This is a handwritten digits predictor. It requires 28x28 input array with 1 and 0 
    representing black and white pixels. 
    """
    input_pixels_cols = 28
    input_pixels_rows = 28
    output_layer_size = 10
    output_classes_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    def __init__(self, weights: str):
        """ Initialization of the perceptron.
        :param weights: Path to the weights to be used for the predictions
        """
        self.retina_layer_size = self.input_pixels_cols * self.input_pixels_rows
        super(HandWrittenDigitPredictor, self).__init__(self.retina_layer_size, self.output_layer_size, self.output_classes_labels, weights)

    def _move_center_of_mass_to_center(self, stimuli: list):
        """ This is part of the input pre-process. It consists on moving the center of mass of the handwritten
        digit to the center of the 28x28 frame
        :param stimuli: 1-dimension list of 1 and 0 representing the handwritten digit. 1's are black pixels and 0's the white ones
        """
        np_image = numpy.array(stimuli).reshape(self.input_pixels_cols, self.input_pixels_rows)
        center_of_mass = scipy.ndimage.measurements.center_of_mass(np_image)
        y_delta = 14-int(center_of_mass[0])
        x_delta = 14-int(center_of_mass[1])

        # Apply correction
        corrected_np_image = scipy.ndimage.interpolation.shift(np_image, [y_delta, x_delta], cval=0)
        return list(corrected_np_image.reshape(self.input_pixels_rows * self.input_pixels_cols))

    def _rescale_to_20x20(self, stimuli):
        """ TODO: Reescale the digit to a 20x20 pixels digit """
        pass

    def predict(self, stimuli: list) -> str:
        """ Performs the prediction from the stimuli
        :param stimuli: 1-dimension list of 1 and 0 representing the handwritten digit. 1's are black pixels and 0's the white ones
        """
        # First preprocess the input data
        # stimuli = _rescale_to_20x20(stimuli)
        stimuli = self._move_center_of_mass_to_center(stimuli)
        return super(HandWrittenDigitPredictor, self).predict(stimuli)


class HandWrittenDigitPredictorGUI(HandWrittenDigitPredictor):
    """ 
    This is the GUI for the handwritten digit predictor. It consists on a canvas where to write the digit with the mouse
    For perform the prediction the user must double-click on the canvas. Additionally, the user would be able to load test
    digits by clicking on the mouse right button
    """
    def __init__(self, pixel_size: int, weights: str, test_data_file: str, test_labels_file: str):
        """ Initialization of the class
        :param pixel_size: The actual number of pixels, every one of the 28x28 input pixels would have for width and height
        :param weights: The weights to be used by the handwritten digits predictor
        :param test_data_file: The path to the test images file. It is used to load the test digits
        :param test_labels_file: The path to the test images labels file. It is used to load the test digits
        """
        self.input_pixels_rows = 28
        self.input_pixels_cols = 28
        self.pixel_size = pixel_size
        self.test_data_file = test_data_file
        self.test_labels_file = test_labels_file

        self.canvas_width = self.input_pixels_cols * self.pixel_size
        self.canvas_height = self.input_pixels_rows * self.pixel_size
        master = Tk()
        self.w = Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")

        # Shadow on the border pixels. The original black and white (bilevel) images from NIST were size normalized to 
        # fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a 
        # result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 
        # image by computing the center of mass of the pixels, and translating the image so as to position this point 
        # at the center of the 28x28 field.
        for col in range(self.input_pixels_cols):
            for row in range(self.input_pixels_rows):
                if col < 4 or col >= 24 or row < 4 or row >= 24:
                    self.w.create_rectangle(col * self.pixel_size, 
                                       row * self.pixel_size, 
                                       col * self.pixel_size + self.pixel_size,
                                       row * self.pixel_size + self.pixel_size, fill="#ddd", outline="")

        # Draw the grid
        for col in range(self.input_pixels_rows):
            col_pixel = col * self.pixel_size
            self.w.create_line(col_pixel, 0 , col_pixel, self.canvas_height, dash=(4,2), fill="grey")
        for row in range(self.input_pixels_cols):
            row_pixel = row * self.pixel_size
            self.w.create_line(0, row_pixel, self.canvas_height, row_pixel, dash=(4,2), fill="grey")

        # Clicking the right mouse button you can load test dataset handwritten characters
        self.ubyte_test_dataset = UbtyeDataset(self.test_data_file, test_labels_file)

        self.drawn_rectangles = []  # This is a list with the ID of the drawn rectangles in the canvas
        self.drawn_number = [0 for _ in range(28*28)]  # Initialization of the input of the perceptron

        self.w.bind("<B1-Motion>", self._draw)
        self.w.bind("<Double-Button-1>", self._perform_prediction)
        self.w.bind("<Button-3>", self._load_next_test_number)
        self.w.pack()

        self.statusbar = Label(master, text="Draw a figure!", font=("Courier", 30))
        self.statusbar.pack()

        super(HandWrittenDigitPredictorGUI, self).__init__(weights)

    def _draw(self, event):
        """ Handles the click and drag event on the canvas, It draws a black rectangle where the mouse being passed """
        if event.x >= self.canvas_width or event.x < 0:
            return
        if event.y >= self.canvas_height or event.y < 0:
            return
        pixel_x = int(event.x / self.pixel_size)
        pixel_y = int(event.y / self.pixel_size)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx + pixel_x < 0 or dx + pixel_x >= self.input_pixels_cols:
                    continue
                if dy + pixel_y < 0 or dy + pixel_y >= self.input_pixels_rows:
                    continue
                rid = self.w.create_rectangle((dx+pixel_x) * self.pixel_size, 
                                              (dy+pixel_y) * self.pixel_size, 
                                              (dx+pixel_x) * self.pixel_size + self.pixel_size, 
                                              (dy+pixel_y) * self.pixel_size + self.pixel_size, fill="black")
                self.drawn_rectangles.append(rid)
                self.drawn_number[(dy+pixel_y)*28+(dx+pixel_x)] = 1

    def _perform_prediction(self, event):
        """ Performs the prediction on the number handwritten in the canvas """
        if not any(self.drawn_number):
            self.statusbar.config(text=f"Come on! draw something...")
            return
        prediction = self.predict(self.drawn_number)
        self._clean_canvas()
        self.drawn_number = [0 for _ in range(28*28)]  # Initialization of the input of the perceptron
        self.statusbar.config(text=f"The prediction is: {prediction}")

    def _load_next_test_number(self, event):
        """ Loads the next handwritten number from the test dataset """
        image, label = next(self.ubyte_test_dataset)
        self.drawn_number = [ 1 if image[i] > 0 else 0 for i in range(self.input_pixels_cols * self.input_pixels_rows) ]
        self._draw_number(self.drawn_number)
        self.statusbar.config(text=f"Number from dataset: {label}")

    def _draw_number(self, number):
        """ Cleans the canvas and draws the number passed by parameter """
        self._clean_canvas()
        for col in range(self.input_pixels_cols):
            for row in range(self.input_pixels_rows):
                if not number[self.input_pixels_rows*row + col]:
                    continue
                self.drawn_rectangles.append(
                        self.w.create_rectangle(col * self.pixel_size, 
                                                row * self.pixel_size, 
                                                col * self.pixel_size + self.pixel_size,
                                                row * self.pixel_size + self.pixel_size, fill="black"))

    def _clean_canvas(self):
        """ Clean the canvas """
        for rid in self.drawn_rectangles:
            self.w.delete(rid)
        self.drawn_rectangles.clear()


if __name__ == "__main__":
    """
    For this first experiment I would like to have the perceptron classifying handwritten numbers
    For that end I am using the MNIST database: yann.lecun.com/exdb/mnist
    """

    parser = argparse.ArgumentParser(description="Naive implementation of a perceptron for handwritten digits recognition")
    parser.add_argument("--train", action="store_true", help="Whether to train the model or not (MNIST db). The paths are hardcoded")
    parser.add_argument("--weights", type=str, help="The weights to use")
    args = parser.parse_args()

    # The training is done using the MNIST dataset that can be found here: yann.lecun.com/exdb/mnist
    train_data_file = "data/training_set/train-images.idx3-ubyte"
    train_labels_file = "data/training_set/train-labels.idx1-ubyte"
    test_data_file = "data/test_set/t10k-images.idx3-ubyte"
    test_labels_file = "data/test_set/t10k-labels.idx1-ubyte"
    digit_predictor = HandWrittenDigitPredictorGUI(20, args.weights, test_data_file, test_labels_file)

    if args.train:
        digit_predictor.train(train_data_file, train_labels_file, test_data_file, test_labels_file)
    else:
        if not args.weights:
            raise Exception("When prediction, weights are needed. Please use the --weights flag")
    mainloop()
