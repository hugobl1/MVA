## Object recognition and computer vision 2021/2022

We could find in this file the implementation to practice ensemble learning (see model)
And also the basis of a semi-supervised approch using self-learning (you will need NABirds dataset 
that is not include right there to test it)
The idee is to iteratively train the network and to add the most-confident predictions to the 
dataset thanks to add (that pick the best predictions) and create_new_dataset (that create the new
dataset with the new labeled data)