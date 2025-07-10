---
title: To Descend or Not to Descend? Gradient Descent for Dummies
date: 2025-07-08
categories:
  - Machine Learning
  - Algorithms
  - Gradient Descent
  - Simulation
  - Physics
  - Math
description: How does gradient descent iteratively discover local minimas? How exactly was this algorithm invented from the ground up? What are the limitations of traditional gradient descent?
---
# A Motivation for Gradient Descent
Imagine you are training a simple machine at a basic task: throwing darts on a board. The objective is to land darts as close to the center (the bullseye) as possible. However, the machine has no spatial awareness; it only knows the angle at which it threw the dart and where it landed. How can we design an algorithm that helps the machine discover the optimal throwing angle?

We recall a powerful idea learned from high-school calculus: optimization. To formalize this, we introduce a loss function $L(\theta)$ that measures the error of our machine’s performance. The farther the dart lands from the center, the higher the loss. Minimizing $L(\theta)$ will yield throws closer to the bullseye.

## Optimization, and Its Subtle Challenge
In calculus, optimization locates extrema of a function by analyzing the derivative (or "gradient" for higher dimensional functions). Points where the derivative changes sign indicate local maxima or minima. In formal mathematical language, a point $y$ is a local minima of a function $f$ over a domain $A$ if:
$$\forall x\in A, |x-y|<\delta\implies f(y)\geq f(x)$$
For some neighborhood radius $\delta$. An intuitive and physical analogy of optimizing is to imagine yourself as a raindrop guided by gravity to glide across the surface of the function, until you finally settle in a puddle in the surface. Note, this doesn't have to be the deepest puddle ever (i.e. the global minimum), as long as you are trapped in a ditch, no matter how small, you {{< sidenote "stay there" >}}This might not be the case of Stochastic Gradient Descent{{< /sidenote >}} . 

The challenge that comes with function optimizing however, is that the loss function is often complex (especially in deep learning), and difficult to find a closed, algebraic solution to 
$\text{arg} \min_{\theta}L(\theta)$. The main three factors are due to:
1. **High Dimensionality:** Modern deep learning models tend to have upwards of hundreds of millions and billions of parameters. Mapping out all the combinations of these parameters and discovering a closed form equation for all such parameters is mathematically infeasible. In addition, we would have a system of countless non-linear equations with many variables. Just too complicated.
2. **Critical Values:** Even if it were possible to discover a closed form equation of our model, with all its parameters, sorting through the critical values is another story. Simply put, critical values can be understood as points where the second partial derivative of a particular function is 0 or undefined. We would then have to sort through every single critical value (which there can be billions!) and categorize them into maxima, minima, and saddle points. We would then need to arduously discover what the global minimum is.
3. **Symbolic Differentiation:** Implementing symbolic differentiation on an arbitrary function, is surprisingly a very compute-heavy task!
With these limitations in mind. It's crucial for us to find an alternative approach to discovering minima. This presents an exigency for gradient descent.

# Gradient Descent
{{< admonition type="tip" title="Gradient Descent">}}
An iterative approach to discovering the minima of a function, by moving in the opposite direction of the gradient.
{{< /admonition >}}

In multivariate calculus, a gradient is just a vector containing all the {{< sidenote "partial derivatives" >}}The derivative of a function with respect to a particular variable, with all variables deemed constant. Imagine projecting the function to a 2-dimensional space parallel to the axis we are differentiating with respect to. We then calculate the slope of the resulting projection, and that is our partial derivative.{{< /sidenote >}}. An important property of the gradient, is that it always marks the direction of the greatest increase. {{< backlink "gradient-maximal-increasing-direction-proof" "Do you see why?"  >}} . 

Hence, by calculating the gradient at a certain point, we can follow the opposing direction of the gradient vector to find the direction of the greatest decrease in our function. Back to our rain-water analogy, this value tells us the direction that we must travel in to ensure the greatest possible decrease in loss. All that's left is to decide how much we want to move in this direction, and when to stop descending.

## Towards an Implementation of Gradient Descent.
Knowing a general idea of gradient descent, let's try to implement this algorithm from scratch via our intuition. We will use a simplified example of our dart throwing problem, and assume that the loss function of our dart follows a quadratic function, ranging from $-90^{\circ}$ to $90^{\circ}$. We also stipulate that we throw the dart at a height of $1$ meter, that our board is $10$ meters away, the exit velocity of the machine's dart is $10$ meters per second, and that air resistance is negligible (cue the sighs of relief from the physics students). We may model this system with kinematics, and determine that the optimal angle to hit the target is at $39.261^{\circ}$:

Initial Conditions
$$ v= 10 $$
$$ target_x = 10 $$
$$ target_y = 1 $$
System:
$$ t = \frac{target_x}{v\cdot \cos(\theta)}$$
$$1 + v\cdot\sin(\theta)\cdot t + \frac{1}{2}gt^2=target_y$$
$$1+target_x\cdot\tan(\theta)+\frac{1}{2}g\left(\frac{target_x}{v\cdot\cos(\theta)}\right)^2 = target_y$$
Solve to get $t= 1.29154s$ and $\theta=39.261$.

To finish the setup for our problem statement, we use the following loss function. Let $$y(\theta)=1+target_x\cdot\tan(\theta)+\frac{1}{2}g\left(\frac{target_x}{v\cdot\cos(\theta)}\right)^2$$and the loss function:
$$L(\theta) = \frac{1}{2}(y(\theta) - 1)^2$$.

## Implementation 1: Constant Descent
We try a very simple approach first. At each learning step, we descend by a {{< sidenote "constant factor $\eta$" >}}This is also called the **learning rate** of our model. It defines how big of a "leap of faith" our model takes while descending. {{< /sidenote >}}. Smart individuals may notice that the choice of the learning rate has a great bearing on the performance of our model. Too small of a learning rate, and we may require many iterations to reach a satisfactory minima, not to mention, our model may be stuck in a extremely small local minima. Too big of a learning rate, and our model could potentially skip past a small, very good minima. We can illustrate this behavior below:

![Learning Rates](/images/demystifying_gradient_descent/learning_rates.jpeg)

The update logic of this algorithm is quite simple:
$$\theta_{k+1}=\theta_{k} -\text{sgn}(\nabla L(\theta_k))\cdot\eta$$
We also specify the number of iterations that our model should take before it stops descending, and optionally, a loss threshold where our model prematurely stops too. Our implementation will return a history of the gradient updates, so we can see how model performance changes over iterations.

{{< admonition type="warning" title="Important" >}} 
Our learning rate is in terms of radians, as we convert our angle in degrees to radians for fitting the gradient descent algorithm
{{< /admonition >}}

```python
def constant_descent(learning_rate=0.001, iterations=10000, threshold = 0.1, initial_theta=0, use_threshold=False):
	grad = radians(initial_theta)
	grad_history = [grad]
	for i in range(iterations):
		loss = LOSS_FUNCTION(grad)
		if (loss < threshold and use_threshold):
			print(f"Converged to loss {loss} at angle {np.degrees(grad)} after {i} iterations. Parameters: lr={learning_rate}, threshold={threshold}, initial_theta={initial_theta}")
			return grad_history
		grad_derivative = LOSS_GRADIENT(grad)
		if (grad_derivative > 0):
			grad -= learning_rate
		else:
			grad += learning_rate
		grad_history.append(grad)
	print(f"Arrived at loss {loss} at angle {np.degrees(grad)} after {iterations} iterations. Parameters: lr={learning_rate}, threshold={threshold}, initial_theta={initial_theta}")
	return grad_history
```
Output:
```
Arrived at loss 0.4044264645070036 at angle 57.29577951308232 after 2 iterations. Parameters: lr=0.5, threshold=0.5, initial_theta=0
Arrived at loss 0.01334846874019381 at angle 28.64788975654116 after 4 iterations. Parameters: lr=0.25, threshold=0.1, initial_theta=0
Arrived at loss 0.7332180858751962 at angle 45.836623610465864 after 10 iterations. Parameters: lr=0.2, threshold=0.01, initial_theta=0
Arrived at loss 0.001084857371692168 at angle 34.37746770784939 after 100 iterations. Parameters: lr=0.1, threshold=0.001, initial_theta=0
Arrived at loss 0.00841345475577331 at angle 40.10704565915763 after 1000 iterations. Parameters: lr=0.05, threshold=0.001, initial_theta=0
Arrived at loss 0.00012137736204016537 at angle 38.961130068896004 after 1000 iterations. Parameters: lr=0.01, threshold=0.0001, initial_theta=0
```
From our little experiment, we see that increasing the learning_rate, iterations, and decreasing the threshold allows us to ever-so slightly decrease the loss, and converge closer to our true value of $\theta=39.261$. Let's graph these iterations one by one to see how our gradient descent works:

![Constant GD Performance](/images/demystifying_gradient_descent/constant_gd_trajectories.png)

Some interesting things to note from the plot. We see two identical minima both at $L(\theta) = 0$. This is consistent with our problem statement, as experienced physics students know that projectile problems like this tend to have two solutions! One solution that shoots at some lower angle $\theta_1$, and another solution that shoots at angle $90^{\circ} - \theta_1$. 


Note. Higher $\eta$ speeds up descent but risks overshooting (lr=0.2); smaller $\eta$ improves precision but slows convergence.

Though our current implementation works, we have to struggle and test until we have the perfect parameters to find a given minima. In higher-dimension loss functions, it is difficult to visualize where the minima are, and their exact size. We'd need a more dynamic approach that "slows" down the learning rate once we enter a minima. Enter: the traditional gradient descent.

## Implementation 2: Gradient Descent with Decreasing Learning Rate.
From now on, we will use the following baseline to evaluate our algorithms:
$$ \text{lrs}=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]$$
$$ \text{iterations}=[1, 10, 100, 1000, 10000, 100000]$$
$$\text{thresholds}=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001]$$
Re-evaluating `constant_descent` yields these graphs, which we will use as ground for comparisons:

![Ground](/images/demystifying_gradient_descent/ground.png)

Back to the implementations. Here's an important observation about gradients. The closer we get to the center of a minima, $|\nabla L(\theta)|$ decreases as the "curved surface" flattens out. We use this to reduce our step-size as $\theta_k$ approaches a minimum. This ensures that we overshoot the minima of the function less, the closer we get to an optimized value. Intuitively, it would look like this:

![Decreasing Learning Rates](/images/demystifying_gradient_descent/decreasing_learning_rates.png)

Implementing this is just a small edit to our update function, where we now update $\theta$ with the true gradient
$$\theta_{k+1} = \theta_k - \nabla L(\theta_k)\cdot \eta$$
The implications of this are powerful. At larger gradients greater than 1, our step sizes get amplified so we move further, at smaller gradients lesser than 1, our step sizes get smaller so we gain precision while moving closer to our target value.
```python
def gradient_descent(learning_rate=0.001, iterations=10000, threshold = 0.1, initial_theta=0, use_threshold=False):
	grad = radians(initial_theta)
	grad_history = [grad]
	for i in range(iterations):
		loss = LOSS_FUNCTION(grad)
		if (loss < threshold and use_threshold):
			print(f"Converged to loss {loss} at angle {np.degrees(grad)} after {i} iterations. Parameters: lr={learning_rate}, threshold={threshold}, initial_theta={initial_theta}")
			return grad_history
		grad_derivative = LOSS_GRADIENT(grad)
		# make sure grad is between -90 and 90
		grad -= learning_rate * grad_derivative
		grad = np.clip(grad, radians(-90), radians(90))
		grad_history.append(grad)
	print(f"Arrived at loss {loss} at angle {np.degrees(grad)} after {iterations} iterations. Parameters: lr={learning_rate}, threshold={threshold}, initial_theta={initial_theta}")
	return grad_history
```
Output:
```
Arrived at loss 12.005000000000003 at angle 90.0 after 1 iterations. Parameters: lr=1, threshold=0.001, initial_theta=0
Arrived at loss 8.539623824150093e+65 at angle -90.0 after 10 iterations. Parameters: lr=0.1, threshold=0.001, initial_theta=0
Converged to loss 0.0008027951509844755 at angle 38.59420083394617 after 16 iterations. Parameters: lr=0.01, threshold=0.001, initial_theta=0
Converged to loss 0.000987667255874936 at angle 38.524154886156914 after 188 iterations. Parameters: lr=0.001, threshold=0.001, initial_theta=0
Converged to loss 0.0009982206437995594 at angle 38.52037715723293 after 1902 iterations. Parameters: lr=0.0001, threshold=0.001, initial_theta=0
Converged to loss 0.0009999688373613636 at angle 38.51975343526484 after 19039 iterations. Parameters: lr=1e-05, threshold=0.001, initial_theta=0
```
Immediately, we see that with the same arguments as `constant_descent`, `gradient_descent` converges to our threshold much faster, requiring only $19040$ steps in the lowest learning rate setting compared to the over 67k steps used by `constant_descent`. The plots are added below:

![Decreasing LR Plots](/images/demystifying_gradient_descent/decreasing_lr_plots.png)

This is the current gradient descent implementation used in most machine learning libraries. 
## (Optional) Improvement 3:  Decay
A further improvement on our current implementation would be introducing a decay on our learning-rate on top of using the gradient as a multiplicative factor. {{< sidenote "Why?" >}}Imagine that our gradient some how perfectly lands on a position on the loss surface, where it bounces symmetrically between the sides of the surface without a change to the magnitude of our gradient. The gradient-based factor would no longer be sufficient as our learning rate remains constant. This bouncing behavior would continue indefinitely.{{< /sidenote >}} 

Adding this decay avoids any risk of having an "infinite bounce" in our training procedure. With more steps, the learning rate naturally decreases until smaller steps are taken. This ensures that eventually, our step size becomes small, and avoids any oscillations.

# Closing Words
Congratulations! If you made it this far, you know know how to implement your own gradient descent algorithm! That being said, here are some questions to consider:
1. What if our system was not controlled by one simple variable? Take a neural network with billions of parameters. Is our current implementation still feasible?
2. What are some ideas to scale up the traditional gradient descent algorithm? How do we expand this to multiple dimensions?
3. What makes a good loss function? Are there certain standards used by ML engineers? What are they?
4. How can we efficiently calculate a gradient?
Find out the answer to these and more in the sequel of this blog! Which should hopefully, come very soon.

Although Gradient Descent has a scary name, it's intuition is simple, and it's performance is elegant.

![Meme](/images/demystifying_gradient_descent/meme.png)

Find the full code for this blog at this [notebook](https://github.com/AlexAtBerkeley/alex-berkeley/blob/main/content/notebooks/gradient_descent.ipynb).