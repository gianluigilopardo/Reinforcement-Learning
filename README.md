# Stochastic optimization

Here I present a **Reinforcement learning** application to the **Stochastic optimization** problem which follow.

Let us consider the following system. A small shop has two employees. One is always serving customers, the other is doing other jobs as
well. Suppose that the number of customers in the shop is between 0 and 6, since any further incoming customer would not enter the shop if
6 people are already in. Action 1 correspond to let the second employee do his other jobs and having a single employee serving the customers.
Under this action the system follows the following dynamics. At each unit of time if the current number of customers is between one and 5,
there is equal probability that
* a new customer enter the shop
* nothing happens
* a customer is served and exit the shop

If there is no customer in the shop, with probability 2/3 nothing happens in the next unit of time and with probability 1/3 a new customer
enters the shop. If there are 6 customers in the shop, with probability 2/3 nothing happens in the next unit of time and with probability 1/3 a customer will be served and exit the shop. Rewards are discounted by a factor λ = 0.9. There is an immediate reward of 1 unit for each customer served (whatever the number of customers in queue). If the states are numbered from 1 to 7, and at state i we have i−1 customers in queue, the reward matrix is
```math
      [1 0 0 0 0 0 0]
      [0 1 0 0 0 0 0]
R1 =  [0 0 1 0 0 0 0]
      [0 0 0 1 0 0 0]
      [0 0 0 0 1 0 0]
      [0 0 0 0 0 1 0]
```
Action 2 is instead to have both employees serving the customers. In this case the dynamics would be changed as follows. At each unit of
time if the current number of customers is between one and 5,
* a new customer enter the shop with probability 1/4
* nothing happens with probability 1/4
* a customer is served and exit the shop with probability 1/2

If there is no customer in the shop, with probability 2/3 nothing happens in the next unit of time and with probability 1/3 a new customer enters the shop. If there are 6 customers in the shop, with probability 1/3 nothing happens in the next unit of time and with probability 2/3 a customer will be served and exit the shop. Whit respect to the previous reward structure, under this action there is an additional fixed cost per unit of time of 0.2 units due to the fact that, while serving, the second employee is not doing the other jobs. Therefore the reward matrix is
```math
      [−0.2 −0.2 −0.2 −0.2 −0.2 −0.2 −0.2]
      [ 0.8 −0.2 −0.2 −0.2 −0.2 −0.2 −0.2]
R2 =  [−0.2  0.8 −0.2 −0.2 −0.2 −0.2 −0.2]
      [−0.2 −0.2  0.8 −0.2 −0.2 −0.2 −0.2]
      [−0.2 −0.2 −0.2  0.8 −0.2 −0.2 −0.2]
      [−0.2 −0.2 −0.2 −0.2  0.8 −0.2 −0.2]
      [−0.2 −0.2 −0.2 −0.2 −0.2  0.8 −0.2]
```
Please use Reinforcement Learning to learn the best policy. As a stepsize rule, please use 
```math
αk = A/(B + k)
```
with A = 150 and B = 300. The number of iterations can be 105. Use as initial state the one where no customer is in the shop.
