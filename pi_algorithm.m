function pi_algorithm

%-------------------------------- start -----------------------------------
clear; close all; clc;

global R; global Q;global A; global B;

% information of system 

A = [  0,      1;...
       -1,    1.99   ];

B = [  0;...
       1  ];

x0 = [1;-1];


% action network
actor_middle_num = 15;
actor_epoch = 5000;
actor_err_goal = 1e-9;
actor_lr = 0.1;
actor = newff([-10, 10; -10, 10], [actor_middle_num 1], {'tansig' 'purelin'},'trainlm');
actor.trainParam.epochs = actor_epoch; 
actor.trainParam.goal = actor_err_goal; 
actor.trainParam.show = 10; 
actor.trainParam.lr = actor_lr; 

% critic network
critic_middle_num = 15;
critic_epoch = 10000;
critic_err_goal = 1e-9;
critic_lr = 0.1;
critic = newff([-10, 10; -10, 10], [critic_middle_num 1], {'tansig' 'purelin'},'trainlm');
critic.trainParam.epochs = critic_epoch;
critic.trainParam.goal = critic_err_goal; 
critic.trainParam.show = 10;  
critic.trainParam.lr = critic_lr; 
critic.biasConnect = [1;0];

u = zeros(400);
state_dim = size(A,1);
control_dim = size(B,2);

Q = 1*eye(state_dim);
R = 1*eye(control_dim);
epoch = 15;
eval_step = 400;
performance_index = ones(1,epoch + 1);

x_tr = zeros(state_dim,1);
x_ta = zeros(state_dim,1);
e_train = zeros(state_dim,1);

for i = 1:50
    x_tr = [x_tr, zeros(state_dim,1)];  
    x_tr = [x_tr,2*(rand(state_dim,1)-0.5)]; 
    x_tr = [x_tr,1*(rand(state_dim,1)-0.5)];
    x_tr = [x_tr,0.5*(rand(state_dim,1)-0.5)];

    x_ta = [x_ta, zeros(state_dim,1)];  
    x_ta = [x_ta,4*(rand(state_dim,1)-1)]; 
    x_ta = [x_ta,2*(rand(state_dim,1)-1)];
    x_ta = [x_ta,(rand(state_dim,1)-1)]; 
end

for i = 1:200
    e_train = [e_train, x_ta(:,i) - x_tr(:,i)]
end

figure(1),hold on;
h = waitbar(0,'Please wait');
for i = 1:epoch
    % update critic
    % evaluate policy
    critic_target = evaluate_policy(actor, e_train, x_tr, x_ta, eval_step);
    critic = train(critic,e_train,critic_target);  
    
    performance_index(i) = critic(x0);
    figure(1),plot(i,performance_index(i),'*'),xlim([1 epoch]),hold on;
    
    waitbar(i/epoch,h,['Training controller...',num2str(i/epoch*100),'%']);
    if i == epoch
        break;
    end
    
    % update actor
    actor_target = zeros(control_dim,size(e_train,2));
    for j = 1:size(e_train,2) - 1
        e = e_train(:,j);
        if e == zeros(state_dim,1)
            ud = zeros(control_dim,1);
        else
            objective = @(u) cost_function(e,u) + critic(e_train(:,j+1));
            u0 = actor(e);
            ud = fminunc(objective, u0);
        end
        actor_target(:,j) = ud;
    end   
    actor = train(actor, e_train, actor_target);

end
close(h)

figure(1),
xlabel('Iterations');
ylabel('$V(x_0)$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
hold off;

save training_results/actor_critic actor critic
end


%---------------------------- evaluate policy -----------------------------
function y = evaluate_policy(actor,e,x_tr,x_ta,eval_step)
critic_target = zeros(1,size(e,2));
for k = 1:eval_step
    uep = actor(e);
    critic_target = critic_target +  cost_function(e,uep);
    e = controlled_system(x_tr,x_ta,uep);
end
y = critic_target;
end

%--------------------------- output of system ----------------------------
function y = controlled_system(x_tr,x_ta,u)
global A; global B;
% system matrices
y = A*x_ta + B*u - A*x_tr;  
end

%----------------------------- cost function ------------------------------
function y = cost_function(e,u)
global R; global Q;
y = (diag(e'*Q*e) + diag(u'*R*u))';
end

