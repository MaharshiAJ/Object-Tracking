# Task 1

## Part 1
![Architecture Diagram](Raw%20Diagram%20Data/Deep%20Sort%20Diagram.drawio.svg)


## Part 2
- FrRCNN
    - Feature extraction will extract the features from the convolutional layer and present it to the estimation step
    - Feature extraction stage will also provide regions for the classification stage. This however is not used for the tracking process and so this can be ignored.
- Estimation
    - The state of each model can be described as 7 (or 8) dimensional vector containing the properties,
        - (u,v) representing the bounding box center
        - s representing the area of the bounding box
        - r representing the aspect ratio of the bounding box
        - $\dot{u}$, $\dot{v}$, $\dot{s}$ representing the velocity components
        - Aspect ratio is kept constant however if it is not then the extra variable becomes $\dot{r}$
    - When an object is detected, the state of the object is calculated using a Kalman filter
- Data Association
  - Uses hungarian algorithm to detections to tracks
  - Uses Mahalanobis distance to incorporate motion
- Track management
  - Updates or deletes if a certain object is no longer on screen
  - If an object is deleted and then returns then it returns under a new identity
### Kalman Filter Background
State Equation:
$$x_{k+1} = \Phi x_k + w_k$$
$x_k$ = State vector at a given time k, dimension nx1 \
$\Phi$ = Transition matrix from state at time k to state at time k+1, dimension nxm \
$w_k$ = Associated white noise with a known covariance, dimension nx1 \
Observation Equation:
$$z_k = Hx_k + v_k$$
$z_k$ = Actual measurement of x at time k, dimension mx1 \
H = Noiseless connection between the state vector and measurement vector, dimension mxn \
$v_k$ = Associated measurement error (another white noise process with known covariance), dimension mx1
$$Q = E[w_kw_k^T]$$
$$R = E[v_kv_k^T]$$
Q and R represent the covariances of the noise models shown above
$$P_k = E[(x_k - \hat{x}_k)(x_k - \hat{x})^T]$$
$P_k$ is the error covariance matrix at a given time t \
The term between E is the MSE
### Kalman Filter Equations
$$\hat{x}_k = \hat{x}'_k + K_k(z_k - H\hat{x}_k')$$
$$K_k = \frac{P'_kH^T}{HP'_kH^T + R}$$
$$P_k = (I - K_kH)P'_k$$
$$P_{k+1} = \Phi P_k\Phi^T + Q$$
![Kalman Filter Diagram](Raw%20Diagram%20Data/Kalman%20Filter.png)

### Hungarian Algorithm
Is used to calculate minimal cost from a Bipartite graph For this use, the algorithm will calculate based on IOU's

The process:
1. Build a square matrix of IOU's where each row represents the detection and each column represents the track
2. Take the minimum value in each row and subtract each entry by it. This means the new minimum will be 0 and each row will contain a 0.
3. Repeat step 2 on columns meaning each column will have at least one 0.
4. Draw the minimum number of straight lines that passes through each 0. If the number of lines = the number of rows and columns, then a minimum assignment can be made (skip next step).
5. Find the minimum number that does not have a line going through it and reduce all elements that do not have a line going through it by that number and then add that number to all elements covered by two lines. 
6. Choose n zeros where n = number of rows and columns and each row and column should only have 1 chosen zero. Sum the values of the original matrix where these three chosen zeros are and this is the optimal minimum cost value in the matrix.
7. This optimal cost represents the assignment of Detection to Tracking
   
### Mahalanobis distance
$$d^{(1)} = (d_j - y_i)^TS_iT-1(d_j - y_i)$$
$d^{(1)}$ = How many standard deviations the detection is away from the mean track location
$$b^{(1)}_{i, j} = \mathbb{1}[d^{(1)}(i,j) \leq t^(1)$$
$b^{(1)}_{i, j}$ = Evaluates to 1 if the association is admissible. 0
$$d^{(2)} = min\{1 - r_j^Tr_k^{(i)}|r_k^{(i)} \in R_i\}$$
$d^{(2)}$ = The smallest cosine distance between the i-th track and the j-th detection in appearance space
$$b^{(2)}_{i, j} = \mathbb{1}[d^{(2)}(i,j) \leq t^(2)$$
$b^{(2)}_{i, j}$ = Evaluates to 1 if the association is admissible else 0.
$$c_{i, j} = \lambda d^{(1)}(i, j) + (1 - \lambda)d^{(1)}(i, j)$$
$c_{i, j}$ = The weighted sum of the two metrics
$$b_{i,j} = \prod^2_m=1b_{i,j}^m$$
If the weighted sum is between the above region then the association is considered admissible. 

## References
Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple Online and Realtime Tracking. 2016 IEEE International Conference on Image Processing (ICIP), 3464–3468. https://doi.org/10.1109/ICIP.2016.7533003

Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric (arXiv:1703.07402). arXiv. http://arxiv.org/abs/1703.07402

Kleeman, L. (n.d.). Understanding and Applying Kalman Filtering. https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/kleeman_understanding_kalman.pdf

Lacey, T. (n.d.). Tutorial: The Kalman Filter. https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf

The munkres assignment algorithm(Hungarian algorithm). (n.d.). Retrieved April 13, 2023, from https://www.youtube.com/watch?v=cQ5MsiGaDY8
