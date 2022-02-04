# Image Classifier for clothing patterns

### Data:
the data was forked from here https://github.com/lstearns86/clothing-pattern-dataset

## Implemented:
  - Pre-processing. Basically get all the images and convert it to a data frame
  - First model using MLP (Multilayer Perceptron)
  - Function to measure accuracy of MLP
  - Confusion matrix
  - Use numbers instead of strings for the classes to improve efficiency
  - Second model in tensorflow
  - plots of tensorflow accuracy score and loss

## To be implemented:
  - Cross validation

## Tecnologies:
- Opencv
- Sklearn
- numpy
- pandas
- tensorflow

## Results:

### First run: simple MLP with 3 hidden layers with 50 neurons each, using black and white version of the images

The first run was made for comparison purposes. I wanted to know how a very simple model compares with a more refined one.

accuracy per class:

  dotted: 0.444444444444444 <br>
  solid: 0.4166666666666667 <br>
  striped: 0.1 <br>
  checkered: 0.16 <br>
  floral: 0.5161290322580645 <br>
  zig zag: 0.08695652173913043 <br>
  
### Second run with tensorflow:

  Results can be seem on the plots bellow. I am really cetic about the result but could find nothing wrong with the code.

![loss plot](/version-two/loss_evolution.png)

![accuracy plot](/version-two/accuracy_evolution.png)

