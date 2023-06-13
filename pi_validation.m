clear; close all; clc;


load training_results/actor_critic.mat


A = [  0,      1;...
       -1,    1.99   ];

B = [  1;...
       1  ];

state_dim = size(A,1);
control_dim = size(B,2);

Q = 1*eye(state_dim);
R = 1*eye(control_dim);
   
x0 = [1;5];
x1 = [1;2];

x = x0;
x_net = x1;
x_base = x1;
xx = x;
e = x1 - x0;
xx_net = x_net;
xx_base = x_net;
uu_opt = [];
uu_net = [];

Jreal = 0;

Fsamples = 300;
h = waitbar(0,'Please wait');
for k = 1:Fsamples
    x = A*x;
    xx = [xx x];
    u_net = sim(actor,e);
    Jreal = Jreal + e'*Q*e + u_net'*R*u_net;
    x_base = A*x_base;
    x_net = A*x_net + B*u_net;
    e = x_net - x;
    xx_net = [xx_net x_net];
    xx_base = [xx_base x_base];
    uu_net = [uu_net u_net];
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h)


figure,
plot(0:Fsamples,xx,'b-',0:Fsamples,xx_net,'r--',0:Fsamples,xx_base,'g--','linewidth',1)
xlabel('Time steps');
ylabel('States'); 
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
figure,
plot(0:Fsamples-1,uu_net,'r--','linewidth',1)
xlabel('Time steps');
ylabel('Control');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;



