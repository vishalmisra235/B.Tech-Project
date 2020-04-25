# B.Tech-Project
Metamorphic Testing Automation Factory

# Metamorphic Testing
Metamorphic Testing is one of those testing techniques through which we can reform ML testing and mitigate the oracle problem. The idea is simple: even if we do not know the correct output of a single input, we might still know the relations between the outputs of multiple inputs, especially if the inputs are themselves related.

This tool will help to automatically test your image classifier on some set of metamorphic relations.

This tool is under testing phase for now. Will release our first version soon.

# Install
Run this command in your ubuntu terminal
pip install -i https://test.pypi.org/simple/ metamorphic-tool

# How to run?
After installing this tool, run the following command in your python terminal:
import predefined_relations.outliers_mr 

After then enter the following inputs:
1. Path to your saved CNN model.
2. Path to your dataset.
3. Shape of the images, height and width.

Dataset should consists of folders containing different category images and should be labeled with 0,1,2,3....

The metamorphic relations used were taken from the following paper:-
# validating a deep learning framework by metamorphic testing.

Link:- 
https://ieeexplore.ieee.org/abstract/document/7961649/
