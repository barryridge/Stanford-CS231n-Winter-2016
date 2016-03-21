import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count += 1
        dW[:,j] += X[i,:].T
    dW[:,y[i]] -= count * X[i,:].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
 
  scores = X.dot(W)

  #
  # It took me forever to figure this out, but the Matlab equivalent of
  # the following is:
  #
  # scores(sub2ind(size(scores), 1:size(scores,1), y+1))
  # 
  correct_class_scores = scores[np.arange(num_train), y]

  margins = (scores.T - correct_class_scores + 1)

  # Find the losses, i.e. the cases where the classifier
  # score exceeded the correct classifier score by a 
  losses = np.maximum(0, margins)
  losses[y, range(num_train)] = 0

  loss = np.sum(losses)
  loss /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  correct_indices = np.zeros(margins.shape)
  correct_indices[y, np.arange(num_train)] = 1

  incorrect_indices = np.ones(margins.shape)
  incorrect_indices[correct_indices == 1] = 0

  correct_sums = np.zeros(margins.shape)
  correct_sums[y, np.arange(num_train)] = np.sum(incorrect_indices * (margins>0), axis=0)

  dW -= correct_sums.dot(X).T
  dW += (incorrect_indices * (margins>0)).dot(X).T

  dW /= num_train

  dW += reg * W

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
