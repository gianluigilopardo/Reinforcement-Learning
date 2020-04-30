% REINFORCEMENT LEARNING.

clear all
close all
clc

% The number of possibile states: from 0 to 6 customers:
states = 7; 
actions = 2; % The number of possibile actions.

% Transition probability matrix under action 1:
P1 = [2/3 1/3 0   0   0   0   0;
      1/3 1/3 1/3 0   0   0   0;
      0   1/3 1/3 1/3 0   0   0;
      0   0   1/3 1/3 1/3 0   0;
      0   0   0   1/3 1/3 1/3 0;
      0   0   0   0   1/3 1/3 1/3;
      0   0   0   0   0   1/3 2/3];
 
% Transition probability matrix under action 2:
P2 = [2/3 1/3 0   0   0   0   0;
      1/2 1/4 1/4 0   0   0   0;
      0   1/2 1/4 1/4 0   0   0;
      0   0   1/2 1/4 1/4 0   0;
      0   0   0   1/2 1/4 1/4 0;
      0   0   0   0   1/2 1/4 1/4;
      0   0   0   0   0   2/3 1/3];

% Transition reward matrix under action 1:
R1 = [0 0 0 0 0 0 0;
      1 0 0 0 0 0 0;
      0 1 0 0 0 0 0;
      0 0 1 0 0 0 0;
      0 0 0 1 0 0 0;
      0 0 0 0 1 0 0;
      0 0 0 0 0 1 0];
      
% Transition reward matrix under action 2:
R2 = [-0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2;
       0.8 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2;
      -0.2  0.8 -0.2 -0.2 -0.2 -0.2 -0.2;
      -0.2 -0.2  0.8 -0.2 -0.2 -0.2 -0.2;
      -0.2 -0.2 -0.2  0.8 -0.2 -0.2 -0.2;
      -0.2 -0.2 -0.2 -0.2  0.8 -0.2 -0.2;
      -0.2 -0.2 -0.2 -0.2 -0.2  0.8 -0.2];

% The matrix Q is initialized with all entries equal to 0:
Q = zeros(states,actions); 
lambda = 0.9; % The discount factor.
kmax = 1e+5; % The maximum number of iterations.
A = 150;
B = 300;
% The initial state is 1, which means 0 customers in the shop:
state_current = 1;

% NOTE: in order to make the code easily understandable, in the  
% following comments "i" will indicate the current state and "j" 
% will indicate the successive state. What is more, the notation 
% of the textbook is used, for example p(i,a,j) will indicate 
% the transition probabily from i to j under the action a.

for k = 1:kmax
   alpha = A/(B+k); % The step-size is updated at each iteration.
   
   % The following variabiles represent respectively the
   % probability to move from the current state i to i+1, i-1 and  
   % to remain in i: here they are initialized equal to 0.
    pp1 = 0; % = p(i,i+1)
    pm1 = 0; % = p(i,i-1)
    pss = 0; % = p(i,i)
    
   % The action is chosen randomly as a Bernoulli distribution 
   % with parameter 1/2:
   a = binornd(1,0.5); % It indicates the action chosen: 
   % 0 means action 1, 1 means action 2. It will be helpful later.
   action = a + 1; % The action taken.
   
   % pp1 = p(i,i+1): probability a new customer arrives:
   if(state_current ~= 7) % If #customers=6 -> pp1=0.
       pp1 = (1-a)*P1(state_current,state_current+1)+...
       a*P2(state_current,state_current+1);
   end
   % pm1 = p(i,i-1): probability a customer is served and leave:    
   if(state_current ~= 1) % if #customers=0 -> pm1=0.
       pm1 = (1-a)*P1(state_current,state_current-1)+...
       a*P2(state_current,state_current-1);
   end
   % pss = p(i,i): probability nothing happens:
   pss = (1-a)*P1(state_current,state_current)+...
       a*P2(state_current,state_current);

   % I need to simulate a process that goes to the (i+1)th state 
   % with probability pp1, remains in the same state with 
   % probability pss and goes to the (i-1)th state with 
   % probability pm1.
   % I proceed in that way:
   % 1) Pick a uniformly distributed number u between 0 and 1.
   %    [Remeber that the cumulative distribution function  
   %    (CDF) of a U(a,b) is F(x)=(x-a)/(b-a), if x in (a,b).
   %    In our case F(x)=x, if x in (0,1)]
   % 2) Partition the CDF in 3 parts: from 0 to pp1 (a), from pp1 
   %    to pp1+pss (b) and from pp1+pss to pp1+pp2+pss=1 (c).
   % 3) Choose the states i+1 if u is in the part (a), i if 
   %    u is in the part(b) and i-1 if u is in the part (c).
   
   u = rand(1);

   if(u <= pp1  && state_current ~= 7)
       state_next = state_current+1;
   end
   if(u > pp1 && u <= pp1+pss)
       state_next = state_current;
   end
   if(u > pp1+pss && u <= 1 && state_current ~= 1)
       state_next = state_current-1;
   end
   
   % The reward obtained from going to state_current to state_next
   % is equal to r(i,1,j) if the action taken is 1, while it is
   % equal to r(i,2,j) if the action taken is 2.
   r = (1-a)*R1(state_current,state_next)+...
       a*R2(state_current,state_next);
   
   Qmax = max(Q(state_next,:));
   Q(state_current,action) = (1-alpha)*Q(state_current,action)+...
       alpha*(r+lambda*Qmax);
   
   % At the end the current state is updated:
   state_current = state_next;
end

% The best policy vector d is calculated:
for state = 1:states
   if(Q(state,1) > Q(state,2))
       d(state) = 1;
   else d(state) = 2;
   end
end

% Data visualization:
[Q d']
stat=0:6;
plot(stat,Q(:,1),'r',stat,Q(:,2),'b',stat,Q(:,1),'ro',...
    stat,Q(:,2),'bo','linewidth',2)
title('Q-factors behaviour')
xlabel('Number of customers in the shop') 
legend('Under Action 1','Under Action 2','Location','southeast')
ax = gca;
ax.FontSize = 15;
grid on
