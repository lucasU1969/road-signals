# Traffic signs

Traffic signs is a personal project created to practice fine tuning for pre-trained models. The idea was to understand how to re-train a computer vision model with different classes. 

Some conclusions of this project: 
- The amount of epochs needed for the fine tuning is proportional to the difference between the classes that the model identifies and the new classes. If the classes are similiar, less epochs are needed.
- The lr is also proportional to the difference between the previous and new classes. Also a small lr causes very small changes in the loss function and with a bigger lr the result tends to oscilate. 
