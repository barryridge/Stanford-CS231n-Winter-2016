import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)

    # Normalization trick to resolve numerical instability
    # when dealing with the large exponential terms.
    scores -= np.max(scores)

    # Cache some terms that are used repeatedly.
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)
    correct_class_score = scores[y[i]]
    
    # Update the loss 
    loss -= correct_class_score
    loss += np.log(sum_exp_scores)

    # Update the gradient
    dW[:,y[i]] -= X[i,:].T
    for j in xrange(num_classes):
      dW[:,j] += ((X[i,:].T * exp_scores[j]) / sum_exp_scores)

  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg*W
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Calculate scores for each classifier (column in the weight matrix W)
  # acting on each training sample (row in X)
  scores = X.dot(W)
    
  # Normalization trick to resolve numerical instability
  # when dealing with the large exponential terms.
  scores -= np.max(scores)

  # Cache some terms that are used repeatedly.
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores,axis=1)

  # Find the correct classifier scores for each training sample
  correct_class_scores = scores[np.arange(num_train), y]

  # Update the loss
  loss = np.sum(-correct_class_scores + np.log(sum_exp_scores))

  # Update the gradient
  correct_indices = np.zeros(scores.shape)
  correct_indices[np.arange(num_train), y] = 1

  dW -= correct_indices.T.dot(X).T
  dW += X.T.dot((exp_scores.T / sum_exp_scores).T)
  
  # Average over the training samples
  loss /= num_train
  dW /= num_train
  
  # Add regularization.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

