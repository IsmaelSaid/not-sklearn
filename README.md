# Machine learning algorithms

def descente_gradient_vector (x,y,theta_innit, n_iterations, taux_apprentissage):
  y = y.reshape(y.shape[0],1)
  X = np.hstack((x,np.ones(len(y)).reshape(100,1)))

  def h(theta,X):
    return X.dot(theta)

  def cost_function(theta,X,y):
    return 1/len(x) * np.sum((y-h(theta,X)**2))


  def gradient(theta,X,y):
    return (1/len(y)) * X.T.dot(X.dot(theta) - y)

  def descente_gradient(theta,X,y,n_iterations,alpha):
    next = theta 
    for i in range(1, n_iterations):
      next =  next - alpha * gradient(next,X,y)
    return next

  solution = descente_gradient(theta_innit,X,y,n_iterations,taux_apprentissage)
  return solution
  
print(descente_gradient_vector(x,y,np.random.rand(2,1),1000,0.1))

