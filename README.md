# Apollo
## Making machine learning more accessible for everyone; a wrapper for `sklearn`

### What is Apollo all about?
Have you wanted to learn how to use machine learning in your programs? Maybe you're an advanced user, seeking to add a breath of AI to your million dollar software idea. Maybe you're a novice user who is trying to learn programming so they can make Skynet. But you took one look at the 80+ lines of mysterious sklearn and TensorFlow code it takes to do the simplest classification and recoiled.

You have a dataset. You want to train a network, ask it about something, and have it predict what's going to happen. Your code is already complicated enough without adding hundreds of lines every loop. Why should the simplest case be so complicated?

### Example usage

That's where Apollo comes in. Remember the miles of code? Gone. Here's an example of Apollo's usage:


	# test of the classification and data formatting module

	from apollo_ml import DataFormatterv1
	from apollo_ml import ClassificationNetv1

	# put the table into a PANDAS dataset object
	dataset = DataFormatterv1.Format("../../training_sets/FormulaicDataset.csv")

	print ClassificationNetv1.Predict(dataset, 3, [-45, 2, -7])


And that's it. These few lines read the dataset, train a network, and try your prediction. 

### Looking forward
What's next? These are features I plan on implementing in Apollo:

- I want Apollo to have a GUI that can do the most basic things
- I want Apollo read from a configuration file so more advanced users can have more say over whats going on
- I want to expand the possibilities of `ClassificationNetv1` so it can be used for basic prediction as well as classification

