

%WARNING WORKSPACE WILL BE CLEARED !!!!!
clc
clear
close all

%Instructions :

%just uncomment one of the following line and run


%% Open points cloud:

load Nefertiti.mat
% load Hypersheet.mat
% load pipes.mat
% load Monkey2.mat
% load Foot.mat
% load HandOliver
% load Falangi.mat
% load Mannequin
% load cactus.mat




%% Run  program
[t]=MyCrustOpen(p);

%% plot the points cloud
figure(1);
set(gcf,'position',[0,0,1280,800]);
subplot(1,2,1)
hold on
axis equal
title('Points Cloud','fontsize',14)
plot3(p(:,1),p(:,2),p(:,3),'g.')
axis vis3d
view(3)


%% plot the output triangulation
figure(1)
subplot(1,2,2)
hold on
title('Output Triangulation','fontsize',14)
axis equal
trisurf(t,p(:,1),p(:,2),p(:,3),'facecolor','c','edgecolor','b')%plot della superficie
axis vis3d
view(3)


fprintf('\n')
fprintf('\n')
disp('This work is free thanks to our sponsors and users graditude:')
fprintf('\n')
disp('<a href="http://www.wordans.com/affiliates/?s=2664&amp;a=904">-WORDANS: Make you own T-Shirt</a>')	
fprintf('\n')
disp('<a href="http://www.odicy.com/affiliates/?s=2662&amp;a=903">-ODICY: affordable luxury made easy</a>')
fprintf('\n')
disp('<a href="http://www.gigasize.com/affiliates/?s=2665&amp;a=902">-GIGASIZE: the easiest way to upload and share files</a>')
fprintf('\n')
disp('<a href="http://www.advancedmcode.org/">-Advanced M-code</a>')
fprintf('\n')
disp('<a href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=8412682">-Donate </a>')
fprintf('\n')
disp('Thank you!')
fprintf('\n')





