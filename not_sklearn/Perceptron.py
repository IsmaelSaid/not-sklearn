import numpy as np
class CustomPerceptronClassifier(): 

  def __init__(self,lr = 0.01,n_iterations = 100):
    """_summary_

    Args:
        lr (float, optional): _description_. Defaults to 0.01.
        n_iterations (int, optional): _description_. Defaults to 100.
    """
    self.lr = lr 
    self.n_iterations = n_iterations 
  
  def sigmoid(self,x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1/(1 + np.exp(-x))

  def predict(self,w,b,x):
    """_summary_

    Args:
        w (_type_): _description_
        b (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    predictions = self.predict_proba(w,b,x)
    predictions[predictions > 0.5] = 1 
    predictions[predictions < 0.5] = 0 
    return predictions 

  def predict_proba(self,w,b,x):
    """_summary_

    Args:
        w (_type_): _description_
        b (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = x.dot(w) + b
    return 1/(1+np.exp(-z))
    
  def cost_function(self,a,x,y):
    """_summary_

    Args:
        a (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = len(x)
    cost = 1/len(y) * np.sum(-y * np.log(a) - (1 - y) *np.log(1-a)) 
    return cost

  def gradient(self,x,y,activation):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        activation (_type_): _description_

    Returns:
        _type_: _description_
    """
    dw = 1/len(y) * np.dot(x.T, activation)
    db = 1/len(y) * np.sum(activation - y)
    return (dw, db)

  def fit(self,x,y):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    w = np.random.randn(x.shape[1],1)
    b = np.random.randn(1)
    losses = []

    for i in range(self.n_iterations):
      a = self.predict_proba(w,b,x)
      dw , db = self.gradient(x,y,a)
      w = w - self.lr * dw 
      b = b - self.lr * db
      new_loss = self.cost_function(a,x,y)
      losses.append(new_loss)
    return (losses, w, b)