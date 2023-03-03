
import numpy as np 
class CustomLinearRegression():
    def __init__(self,lr, n_iterations):
        """_summary_

        Args:
            lr (_type_): _description_
            n_iterations (_type_): _description_
        """
        self.lr = lr 
        self.n_iterations = n_iterations 
        
    def model(self,w,x):
        """_summary_
 
        Args:
            w (_type_): _description_
            x (_type_): _description_

                Returns:
            _type_: _description_
        """
        return np.dot(x,w)
    
    def cost_function(self,x,y,model):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1/len(x) * np.sum(((y-model)**2))
    
    def gradient(self,x,y,model):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (1/len(y)) * x.T.dot(model - y) 
    
    def fit(self, x, y):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        m = len(x)
        losses = []
        w = np.random.randn(x.shape[1],1)
        
        for i in range(self.n_iterations):
            model = self.model(w,x)
            loss = self.cost_function(x,y,model)
            losses.append(loss)
            w = w - self.lr * self.gradient(x,y,model)
        return (losses, w) 
            
    def predict(self,w,x):
        """_summary_

        Args:
            w (_type_): _description_
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.model(w,x)
