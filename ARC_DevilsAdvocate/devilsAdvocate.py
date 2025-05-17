import os
import json
import copy
import random
import math
import itertools
import functools
from collections import defaultdict, Counter
from pprint import pprint

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
%matplotlib inline

from tqdm.notebook import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


import tqdm
from random import sample
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


class ARCDataset:
    def __init__(self, train_path=None, train_solutions_path=None, test_path=None, eval_path=None, eval_solutions_path=None):
        self.train_data = self._load_json(train_path) if train_path else {}
        self.train_solutions = self._load_json(train_solutions_path) if train_solutions_path else {}
        self.test_data = self._load_json(test_path) if test_path else {}
        self.eval_data = self._load_json(eval_path) if eval_path else {}
        self.eval_solutions = self._load_json(eval_solutions_path) if eval_solutions_path else {}

        # Define colormap and normalization
        self.ARC_COLORMAP = colors.ListedColormap([
            '#301934', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#000000'  # 11th color for blank
        ])
        self.ARC_NORM = colors.Normalize(vmin=0, vmax=11)

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_task(self, task_id, split='train'):
        if split == 'train':
            return self.train_data.get(task_id), self.train_solutions.get(task_id)
        elif split == 'test':
            return self.test_data.get(task_id), None
        elif split == 'eval':
            return self.eval_data.get(task_id), self.eval_solutions.get(task_id)
        else:
            raise ValueError("split must be 'train', 'test', or 'eval'")

    def resize_grid_to_30x30(self, grid):
        """
        Resizes a grid to 30x30, adding blank pixels (value 10) as needed,
        and positions the original grid in the bottom-left corner.
        """
        new_grid = [[10] * 30 for _ in range(30)]
        original_height = len(grid)
        original_width = len(grid[0])
        
        for i in range(original_height):
            for j in range(original_width):
                new_grid[30 - original_height + i][j] = grid[i][j]

        return new_grid

    def revert_from_30x30(self, grid):
        """
        Converts a tweaked (30x30) grid back to its original raw format
        by removing blank pixels.
        """
        non_blank_rows = [i for i, row in enumerate(grid) if any(cell != 10 for cell in row)]
        non_blank_cols = [j for j in range(len(grid[0])) if any(row[j] != 10 for row in grid)]
        
        min_row, max_row = min(non_blank_rows), max(non_blank_rows)
        min_col, max_col = min(non_blank_cols), max(non_blank_cols)

        return [row[min_col:max_col + 1] for row in grid[min_row:max_row + 1]]

    def create_tweaked_training_data(self, task_data):
        """
        Creates tweaked (30x30) versions of training inputs and outputs.
        Returns two lists: tweaked_inputs and tweaked_outputs.
        """
        tweaked_inputs = [self.resize_grid_to_30x30(example['input']) for example in task_data.get('train', [])]
        tweaked_outputs = [self.resize_grid_to_30x30(example['output']) for example in task_data.get('train', [])]
        return tweaked_inputs, tweaked_outputs

    def create_tweaked_unsolved_input(self, task_data):
        """
        Creates tweaked (30x30) version of an unsolved task input.
        Returns a single tweaked grid.
        """
        test_examples = task_data.get('test', [])
        if not test_examples:
            raise ValueError("No test examples found in task data.")
        return self.resize_grid_to_30x30(test_examples[0]['input'])

    def plot_raw_task(self, task_data, task_solution, title="Raw Task Visualization"):
        """
        Plots the raw task data (inputs and outputs) along with the task solution in a 3x2 grid layout.
        """
        train_examples = task_data.get('train', [])
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))
        plt.suptitle(title, fontsize=16)

        for col, example in enumerate(train_examples):
            # Row 1: Raw inputs
            axs[0, col].imshow(example['input'], cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[0, col].set_title(f"Raw Input {col + 1}")
            axs[0, col].axis('off')

            # Row 2: Raw outputs
            axs[1, col].imshow(example['output'], cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[1, col].set_title(f"Raw Output {col + 1}")
            axs[1, col].axis('off')

            if(col==0):
                # Row 3: Raw task solution
                axs[2, col].imshow(task_solution[col], cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
                axs[2, col].set_title(f"Raw Input Question {col + 1}")
                axs[2, col].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_tweaked_task(self, task_data, task_solution, title="Tweaked Task Visualization"):
        """
        Plots the tweaked (30x30) version of the task data (inputs and outputs)
        along with the tweaked task solution in a 3x2 grid layout.
        """
        train_examples = task_data.get('train', [])
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))
        plt.suptitle(title, fontsize=16)

        for col, example in enumerate(train_examples):
            # Row 1: Tweaked inputs
            tweaked_input = self.resize_grid_to_30x30(example['input'])
            axs[0, col].imshow(tweaked_input, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[0, col].set_title(f"Tweaked Input {col + 1}")
            axs[0, col].axis('off')

            # Row 2: Tweaked outputs
            tweaked_output = self.resize_grid_to_30x30(example['output'])
            axs[1, col].imshow(tweaked_output, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
            axs[1, col].set_title(f"Tweaked Output {col + 1}")
            axs[1, col].axis('off')

            if(col==0):
                # Row 3: Tweaked solutions
                tweaked_solution = self.resize_grid_to_30x30(task_solution[col])
                axs[2, col].imshow(tweaked_solution, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
                axs[2, col].set_title(f"Tweaked Input Question {col + 1}")
                axs[2, col].axis('off')

        plt.tight_layout()
        plt.show()

    
DATA_PATH = '/kaggle/input/arc-prize-2025'
dataset = ARCDataset(
    train_path=f'{DATA_PATH}/arc-agi_training_challenges.json',
    train_solutions_path=f'{DATA_PATH}/arc-agi_training_solutions.json',
    test_path=f'{DATA_PATH}/arc-agi_test_challenges.json',
    eval_path=f'{DATA_PATH}/arc-agi_evaluation_challenges.json',
    eval_solutions_path=f'{DATA_PATH}/arc-agi_evaluation_solutions.json',
)
task_data, task_solution = dataset.get_task('025d127b', split='train')
dataset.plot_raw_task(task_data, task_solution, title="Raw Task Visualization")

dataset.plot_tweaked_task(task_data, task_solution, title="Tweaked Task Visualization")
def convert_to_dataframes(self, tweaked_inputs, tweaked_outputs, tweaked_test_input):
        """
        Converts grids to Pandas DataFrames without recoloring.
    
        Parameters:
        - tweaked_inputs: List of tweaked input grids (30x30).
        - tweaked_outputs: List of tweaked output grids (30x30).
        - tweaked_test_input: Single tweaked test input grid (30x30).
    
        Returns:
        - List of DataFrames for inputs, outputs, and test input.
        """

        def grid_to_dataframe(grid):
            """
            Converts a grid to a Pandas DataFrame.
            Keeps the original grid structure and color values intact.
    
            Parameters:
            - grid: 2D list representing the grid (30x30).
    
            Returns:
            - DataFrame of the grid.
            """
            return pd.DataFrame(grid, index=range(29, -1, -1), columns=range(30))

        # Convert all grids to DataFrames
        input_dataframes = [grid_to_dataframe(grid) for grid in tweaked_inputs]
        output_dataframes = [grid_to_dataframe(grid) for grid in tweaked_outputs]
        test_input_dataframe = grid_to_dataframe(tweaked_test_input)
    
        return input_dataframes + output_dataframes + [test_input_dataframe]


    def plot_taks_guess(self, guess, task_solution, title="Task Guess Visualization"):
        """
        Plots the guesses against the actual.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 9))
        plt.suptitle(title, fontsize=16)
        tweaked_solution = self.resize_grid_to_30x30(task_solution[col])
        axs[0, 0].imshow(tweaked_solution, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
        axs[0, 0].set_title("Task Solution")
        axs[0, 0].axis('off')
        axs[0, 1].imshow(guess, cmap=self.ARC_COLORMAP, norm=self.ARC_NORM)
        axs[0, 1].set_title("Guessed Solution")
        axs[0, 1].axis('off')

        plt.tight_layout()
        plt.show()


# Step 1: Generate tweaked data
tweaked_inputs, tweaked_outputs = dataset.create_tweaked_training_data(task_data)
tweaked_test_input = dataset.create_tweaked_unsolved_input(task_data)

# Step 2: Convert grids to DataFrames
dataframes = dataset.convert_to_dataframes(tweaked_inputs, tweaked_outputs, tweaked_test_input)

# Access each DataFrame
input_df1 = dataframes[0]
input_df2 = dataframes[1]
output_df1 = dataframes[2]
output_df2 = dataframes[3]
test_input_df = dataframes[4]

# Display the first input DataFrame
print("Input DataFrame 1:")
input_df1


def create_color_pair_dataframes(input_df1, input_df2, output_df1, output_df2, input_formula, output_formula):
    """
    Creates two sets of paired DataFrames (input1-output1, input2-output2) based on pixel color mappings.
    Each cell in the DataFrame contains a tuple (input_color, output_color) based on formulas.

    Parameters:
    - input_df1: DataFrame of the first input grid (30x30).
    - input_df2: DataFrame of the second input grid (30x30).
    - output_df1: DataFrame of the first output grid (30x30).
    - output_df2: DataFrame of the second output grid (30x30).
    - input_formula: Function/formula to determine the pixel index in the input grid.
    - output_formula: Function/formula to determine the pixel index in the output grid.

    Returns:
    - input1_output1_df: DataFrame of paired colors (input_df1, output_df1).
    - input2_output2_df: DataFrame of paired colors (input_df2, output_df2).
    """
    import pandas as pd

    def bound_index(index):
        """
        Ensure the index is within the range of 0 to 29.
        """
        return max(0, min(29, index))

    def generate_pair_dataframe(input_df, output_df, input_formula, output_formula):
        """
        Helper function to create one paired DataFrame for an input-output pair.
        Each cell contains a tuple (input_color, output_color).
        """
        pair_df = pd.DataFrame(index=input_df.index, columns=input_df.columns)

        for i in range(30):  # Rows
            for j in range(30):  # Columns
                # Determine pixel indices for input and output using the provided formulas
                input_pixel_index = (bound_index(input_formula(i, j)[0]), bound_index(input_formula(i, j)[1]))
                output_pixel_index = (bound_index(output_formula(i, j)[0]), bound_index(output_formula(i, j)[1]))

                # Extract colors for the determined indices
                input_color = input_df.at[input_pixel_index[0], input_pixel_index[1]]
                output_color = output_df.at[output_pixel_index[0], output_pixel_index[1]]

                # Assign color pairs as tuples
                pair_df.at[i, j] = (input_color, output_color)

        return pair_df

    # Create the paired DataFrames for input1-output1 and input2-output2
    input1_output1_df = generate_pair_dataframe(input_df1, output_df1, input_formula, output_formula)
    input2_output2_df = generate_pair_dataframe(input_df2, output_df2, input_formula, output_formula)

    return input1_output1_df, input2_output2_df


# Example formulas for pixel indices
input_formula = lambda i, j: (i, j)  # Use pixel (i, j) in input
output_formula = lambda i, j: (i, j)  # Use pixel (i, j+1) in output

# Call the function to create paired DataFrames
input1_output1_df, input2_output2_df = create_color_pair_dataframes(
    input_df1, input_df2, output_df1, output_df2, input_formula, output_formula
)


input1_output1_df

def create_similarity_dataframe(df1, df2):
    """
    Creates a 30x30 DataFrame indicating similarity between two input DataFrames of color pairs.
    Each cell is 1 if the pair (input_color, output_color) matches across both DataFrames, else 0.

    Parameters:
    - df1: First DataFrame containing color pairs (30x30).
    - df2: Second DataFrame containing color pairs (30x30).

    Returns:
    - similarity_df: DataFrame of size 30x30 with 0s and 1s.
    """
    import pandas as pd

    # Initialize an empty 30x30 DataFrame for similarity
    similarity_df = pd.DataFrame(index=df1.index, columns=df1.columns)

    for i in range(30):  # Rows
        for j in range(30):  # Columns
            # Compare the pairs in the two DataFrames
            similarity_df.at[i, j] = 1 if df1.at[i, j] == df2.at[i, j] else 0

    return similarity_df


# Assume df1 and df2 are outputs from the previous function
similarity_df = create_similarity_dataframe(input1_output1_df, input2_output2_df)

# Display the similarity DataFrame
print("Similarity DataFrame:")
similarity_df

def generate_test_output_with_mapping(similarity_matrix, test_input_df, input_df, output_df):
    """
    Generates a new output image for the test input by applying color mappings derived from
    input-output grids wherever the similarity matrix has 1. Pixels with similarity 0
    are set to gray (color 12).

    Parameters:
    - similarity_matrix: DataFrame with similarity values (1s and 0s).
    - test_input_df: DataFrame of the test input grid (30x30).
    - input_df: DataFrame of the training input grid (30x30) used for color mapping.
    - output_df: DataFrame of the training output grid (30x30) used for color mapping.

    Returns:
    - raw_image_df: DataFrame of the resulting raw image with 30x30 pixels.
    """
    import pandas as pd

    # Generate the color mapping from input_df to output_df based on the similarity matrix
    color_mapping = {}
    for i in range(30):
        for j in range(30):
            # If similarity_matrix is 1, add the color pair from input to output
            if similarity_matrix.at[i, j] == 1:
                input_color = input_df.at[i, j]
                output_color = output_df.at[i, j]
                color_mapping[input_color] = output_color

    # Create the new grid for the test output
    new_output_grid = []
    for i in range(30):
        new_row = []
        for j in range(30):
            test_input_color = test_input_df.at[i, j]
            if similarity_matrix.at[i, j] == 1:
                new_row.append(color_mapping.get(test_input_color, test_input_color))  # Map based on color_mapping
            else:
                # Set gray (color 12) for pixels with similarity 0
                new_row.append(12)
        new_output_grid.append(new_row)

    # Convert the grid to a DataFrame
    raw_image_df = pd.DataFrame(new_output_grid, index=range(30), columns=range(30))
    return raw_image_df.reindex(index=raw_image_df.index[::-1])

# Example similarity matrix, test input, and training grids
similarity_matrix = create_similarity_dataframe(input1_output1_df, input2_output2_df)
input_df = input1_output1_df  # Use input1-output1 mapping as example
output_df = output_df1  # Training output grid

# Generate the resulting raw image
raw_image_df = generate_test_output_with_mapping(similarity_matrix, test_input_df, input_df, output_df)
raw_image_df


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def plot_dataframe_with_grey(dataframe, title="Image with Grey Pixels (Color 12)"):
    """
    Plots a 30x30 DataFrame where the 12th color is treated as grey and
    the (0,0) cell is oriented at the bottom-left corner.

    Parameters:
    - dataframe: Pandas DataFrame (30x30) representing the grid.
    - title: Title for the plot.

    Returns:
    - None: Displays the plot.
    """
    # Replace color 12 with NaN for treating it as grey
    
    colorsC = colors.ListedColormap([
            '#FF00FF', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#000000' , '#808080' # 11th color for blank
        ])
    # Plotting the grid
    plt.figure(figsize=(8, 8))
    plt.imshow(dataframe, cmap=colorsC,  norm=colors.Normalize(vmin=0, vmax=12), origin="upper")
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.colorbar(label="Colors")
    plt.show() 


# Call the plotting function
plot_dataframe_with_grey(raw_image_df, title="Sample Raw Image with Grey")