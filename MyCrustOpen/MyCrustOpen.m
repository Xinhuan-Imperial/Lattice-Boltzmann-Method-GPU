% MyCrustOpen
% 
% 
% This version has been developped for open surface with no sharp edges. 
% 
% Differently from crust based algorithm does not ensure a tight
%   triangluation and sometimes self-intersecant triangles are generated,
%   it is also generally slower. The final surface may need some repair
%   work which this utilitie does not offer.
% 
% But there are two great advantages, this one can be applied on any kind
%   of open surface for which the Crust fails, supports not regular surface
%   like the Moebius ribbon, and most of all, the surface can have any kind
%   of holes, open feature shouldn't create problem.
% You can see   the demo models for examples.
% 
% If any problems occurs in execution, or if you found a bug, have a
%   suggestion or question just contact me at:
% 
% giaccariluigi@msn.com
% 
% 
% 
% Here is a simple example:
% 
% load Nefertiti.mat%load input points from mat file
% 
% [t]=MyCrustOpen(p);
% 
% figure(1)
%         hold on title('Output Triangulation','fontsize',14) axis equal
%         trisurf(t,p(:,1),p(:,2),p(:,3),'facecolor','c','edgecolor','b')
% 
% Input:
%              p is a Nx3 array containing the 3D set of points
% Output:
%              t are points id contained in triangles nx3 array .
% 
% See also qhull, voronoin, convhulln, delaunay, delaunay3, tetramesh.
% 
% Author:Giaccari Luigi
% Last Update: 28/01/2009
% Created: 15/4/2008
%
%
% This work is free thanks to our sponsors and users graditude:
% 
% <a href="http://www.wordans.com/affiliates/?s=2664&amp;a=904">-WORDANS: Make you own T-Shirt</a>	
% 
% <a href="http://www.odicy.com/affiliates/?s=2662&amp;a=903">-ODICY: affordable luxury made easy</a>
% 
% <a href="http://www.gigasize.com/affiliates/?s=2665&amp;a=902">-GIGASIZE: the easiest way to upload and share files</a>
% 
% <a href="http://www.advancedmcode.org/">-Advanced M-code</a>
% 
% <a
% href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_i d=8412682">-Donate </a>
% 
% Thank you!
%


function [t]=MyCrustOpen(p)
%error check

if nargin>1
    error('The only input must be the Nx3 array of points');
end

[m,n]=size(p);
if n ~=3
    error('Input 3D points must be stored in a 3D array');
end 
clear m n


%%   Main

starttime=clock;


%add points to the given ones, this is usefull
%to create outside tetraedroms
tic
[p,nshield]=AddShield(p);
fprintf('Added Shield: %4.4f s\n',toc)


%delaunay 3D triangulation
tic
tetr=delaunayn(p);%creating tedraedron
tetr=int32(tetr);%save memory
fprintf('Delaunay Triangulation Time: %4.4f s\n',toc)

%Get connectivity relantionship among tetraedroms
tic
[tetr2t,t2tetr,t]=Connectivity(tetr);
fprintf('Connectivity Time: %4.4f s\n',toc)


%get Circumcircle of tetraedroms
tic
[cctetr,rtetr]=CCTetra(p,tetr);
fprintf('Circumcenters Tetraedroms Time: %4.4f s\n',toc)


%Get  intersection factor
tic
Ifact=IntersectionFactor(tetr2t,cctetr,rtetr);
fprintf('Intersection factor Time: %4.4f s\n',toc)

clear tetr tetr2t cctetr rtetr




%extract the manifold

tic
tkeep=Walking(p,t,Ifact);
fprintf('Walking Time: %4.4f s\n',toc)


%final traingulation
t=t(tkeep & all(t<=size(p,1)-nshield,2),:);%delete shield triangles


tic
t=ManifoldExtraction(t,p);
fprintf('Manifold extraction Time: %4.4f s\n',toc)



time=etime(clock,starttime);
fprintf('Total Time: %4.4f s\n',time)



end






%% Connectivity

function [tetr2t,t2tetr,t]=Connectivity(tetr)
%Finds ptest for delaunay criterion and tetredral connectivity




numt = size(tetr,1);
vect = 1:numt;
t = [tetr(:,[1,2,3]); tetr(:,[2,3,4]); tetr(:,[1,3,4]);tetr(:,[1,2,4])];
[t,j,j] = unique(sort(t,2),'rows');
t2tetr = [j(vect), j(vect+numt), j(vect+2*numt),j(vect+3*numt)];


% triang-to-tetr connectivity
% Each row has two entries corresponding to the triangle numbers
% associated with each edge. Boundary edges have e2t(i,2)=0.
nume = size(t,1);
tetr2t  = zeros(nume,2,'int32');
count= ones(nume,1,'int32');
for k = 1:numt

    for j=1:4
        ce = t2tetr(k,j);
        tetr2t(ce,count(ce)) = k;
        count(ce)=count(ce)+1;
    end

end


end      % connectivity()




%% Walking
function tkeep=Walking(p,t,Ifact)

%Buils a manifolds surface from the survivors triangles


%Migliorie:
%<--------Migliorare la sintassi di TriAngle



% building the etmap

numt = size(t,1);
vect = 1:numt;                                                             % Triangle indices
e = [t(:,[1,2]); t(:,[2,3]); t(:,[3,1])];                                  % Edges - not unique
[e,j,j] = unique(sort(e,2),'rows');                                        % Unique edges
te = [j(vect), j(vect+numt), j(vect+2*numt)];
nume = size(e,1);
e2t  = zeros(nume,2,'int32');

clear vect j
ne=size(e,1);
np=size(p,1);


count=zeros(ne,1,'int32');%numero di triangoli candidati per edge
etmapc=zeros(ne,4,'int32');
for i=1:numt

    i1=te(i,1);
    i2=te(i,2);
    i3=te(i,3);



    etmapc(i1,1+count(i1))=i;
    etmapc(i2,1+count(i2))=i;
    etmapc(i3,1+count(i3))=i;


    count(i1)=count(i1)+1;
    count(i2)=count(i2)+1;
    count(i3)=count(i3)+1;
end

etmap=cell(ne,1);
for i=1:ne

    etmap{i,1}=etmapc(i,1:count(i));

end
clear  etmapc







tkeep=false(numt,1);%all'inizio nessun trinagolo selezionato


%Start the front

%we pick all triangles which are linked to a dual edge, the must be flat , not
%self-intersecant and more they need to have low intersection factor

% [q]=Quality3(p,t);%triangle quality
pchecked=false(np,1);


%building the queue to store edges on front that need to be studied
efront=zeros(nume*2,1,'int32');%exstimate length of the queue

iterf=1;%efront iterator
nf=0;%numbers of edges on front

% figure(4)
% hold on
% axis equal
% title('Ifactor')
% trisurf(t,p(:,1),p(:,2),p(:,3),Ifact);
% colorbar


%we choose The 1%triangles with lowest Ifact
[is,index]=sort(Ifact);

%start the front with good Ifact triangles
for i=1:numt
  

        %triangles on boundary
        t1=index(i);
        if Ifact(t1)>-.8
            break
        end

        if not(any(pchecked(t(t1,:))))  && Ifact(t1)<-.8 
            tkeep(t1)=true;%primo triangolo selezionato
            pchecked(t(t1,:))=true;
            efront(nf+1:nf+3)=te(t1,1:3);
            e2t(te(t1,1:3),1)=t1;
            nf=nf+3;
        end
    
end


% figure(10)
% axis equal
% hold on
% trisurf(t(tkeep,:),p(:,1),p(:,2),p(:,3));

clear pchecked is index
if nf==0
    error('Front do not start please send a report to the author')
end

while iterf<=nf


    k=efront(iterf);%id edge on front

    if e2t(k,2)>0 || e2t(k,1)<1 || count(k)<2 %edge is no more on front or it has no candidates triangles

        iterf=iterf+1;
        continue %skip
    end
  
    %candidate triangles
    idtcandidate=etmap{k,1};
    idtcandidate(idtcandidate==e2t(k,1))=[];%remove the triangle we come from

    if isempty(idtcandidate)
        iterf=iterf+1;
        continue%skip
    end

  
    %Get Search point and Search radius
    
    ttemp=t(e2t(k,1),:);
    etemp=e(k,:);
    pt=ttemp(ttemp~=etemp(1) & ttemp~=etemp(2));%opposite point to the edge we are studying
     [sp,sr]=SearchPoint(p(e(k,1),:),p(e(k,2),:),p(pt,:));


   dist=0;
    idp=0;
    for c=1:length(idtcandidate)
        ttemp=t(idtcandidate(c),:);
        etemp=e(k,:);
        idp=ttemp(ttemp~=etemp(1) & ttemp~=etemp(2));
        dist(c)=sum((sp-p(idp,:)).^2);%dopo provare a eliminare i punti fuori dal search radius

    end

    ind=dist>sr*sr;
    idtcandidate(ind)=[];
    if isempty(idtcandidate)
        iterf=iterf+1;
        continue
    else
        dist(ind)=[];
    end


%     [dist,id]=sort(L(idtcandidate));%sort per cominciare dal più probabile
%       [dist,id]=sort(dist);%sort per cominciare dal più probabile
      [dist,id]=sort(Ifact(idtcandidate));%sort per cominciare dal più probabile
    idtcandidate=idtcandidate(id);

    %Now we analyze candidate triangles
    
    
    %% Check conformity
    for c=1:length(idtcandidate)

        idt=idtcandidate(c);
        conformity=true;%initilize to true


        %loop trough all edges of the triangle
        for ii=1:3
            e1=te(idt,ii);

            if e2t(e1,2)>0 && e1~=k
                %edge with two trianlgle
                conformity=false;
                break
            elseif e2t(e1,1)>0
                
                %the edge has only one triangle let's see if we can add
                %this one
                
                %get points from the triangles
                t1=e2t(e1,1);t2=idt;
                ttemp=t(t1,:);
                etemp=e(e1,:);
                pt1=ttemp(ttemp~=etemp(1) & ttemp~=etemp(2));%terzo id punto
                ttemp=t(t2,:);
                pt2=ttemp(ttemp~=etemp(1) & ttemp~=etemp(2));%terzo id punto

                [alpha]=TriAngle(p(e(e1,1),:),p(e(e1,2),:),p(pt1,:),p(pt2,:));%angle between triangles
                if alpha>.5
                    conformity=false;%the angle between the traingles was to small
                    break
                end
            end


        end


        if conformity
            break %exit loop if a good triangle has been found
        end


    end

    %Did we found a good triangle?
    if conformity





        %update connectivity data for new triangle
        
        idt=idtcandidate(c);%id of new triangle
        
        tkeep(idt)=true;%keep ot

        %update e2t
        e2t(k,2)=idt;
        %update e2t for the others edges 
        for j=1:3
            ide=te(idt,j);
           
            if e2t(ide,1)<1% %Is it the first triangle for the current edge?
                efront(nf+1)=ide;
                nf=nf+1;
                e2t(ide,1)=idt;
            else                     %no, it is the second one
                efront(nf+1)=ide;
                nf=nf+1;
                e2t(ide,2)=idt;
            end
        end
    end



    iterf=iterf+1;%increase queue iterator






end





fprintf('\tFinal size of the queue is %4.0f edges were %4.0f \n',nf,nume)
%




end



%% SearchPoint
function [sp,sr]=SearchPoint(p1,p2,p3)
%Gets a point outside a given edge of triangle that forms an equilater
%triangle.
%P1,P2,P3 are all Nx3 arrays
%This is the orientation for points
%    p1
%   / | \
% p3  |  SP
%   \ | /
%    p2

%NOTA
%perchè il search point sia all'esterno del triangolo
%p1 p2 devono essere il punto del lato in cui generare il search p
v21=p1-p2;
v31=p3-p1;

n1=[v21(:,2).*v31(:,3)-v21(:,3).*v31(:,2),...
    v21(:,3).*v31(:,1)-v21(:,1).*v31(:,3),...
    v21(:,1).*v31(:,2)-v21(:,2).*v31(:,1)];%normale ai triangoli



v31=n1;

cosdir=[v21(:,2).*v31(:,3)-v21(:,3).*v31(:,2),...
    v21(:,3).*v31(:,1)-v21(:,1).*v31(:,3),...
    v21(:,1).*v31(:,2)-v21(:,2).*v31(:,1)];%coseni direttori retta
cosdir=cosdir/norm(cosdir);
pm=(p1+p2)*.5;
lenge=norm(p1-p2);

% r=max([lenge,norm(p1-p3),norm(p3-p2)]);

% sp=pm+(cosdir)*(sqrt(r^2-.25*lenge^2));%Search point
sp=pm+(cosdir)*(lenge*.866);%Search point


%search radius più è grande più saranno i triangoli scandagliati

% sr=(lenge+norm(p1-p3)+norm(p2-p2))/3;
% sr=r;
sr=lenge;
% sr=max([lenge,norm(p1-p3),norm(p3-p2)]);
end

%% TriAngle
function  [alpha]=TriAngle(p1,p2,p3,p4)
%Computes angle between two triangles
v21=p1-p2;
v31=p3-p1;

tnorm1(1)=v21(2)*v31(3)-v21(3)*v31(2);%normali ai triangoli
tnorm1(2)=v21(3)*v31(1)-v21(1)*v31(3);
tnorm1(3)=v21(1)*v31(2)-v21(2)*v31(1);
tnorm1=tnorm1./norm(tnorm1);


v41=p4-p1;
tnorm2(1)=v21(2)*v41(3)-v21(3)*v41(2);%normali ai triangoli
tnorm2(2)=v21(3)*v41(1)-v21(1)*v41(3);
tnorm2(3)=v21(1)*v41(2)-v21(2)*v41(1);
tnorm2=tnorm2./norm(tnorm2);
alpha=tnorm1*tnorm2';

end






%% Circumcenters tetraedroms
function [cc,r]=CCTetra(p,tetr)
%finds circumcenters from a set of tetraedroms

%points of tetraedrom
p1=(p(tetr(:,1),:));
p2=(p(tetr(:,2),:));
p3=(p(tetr(:,3),:));
p4=(p(tetr(:,4),:));

%vectors of tetraedrom edges
v21=p(tetr(:,1),:)-p(tetr(:,2),:);
v31=p(tetr(:,3),:)-p(tetr(:,1),:);
v41=p(tetr(:,4),:)-p(tetr(:,1),:);

%preallocation
cc=zeros(size(tetr,1),3);


%Solve the system using cramer method
d1=sum(v41.*(p1+p4)*.5,2);
d2=sum(v21.*(p1+p2)*.5,2);
d3=sum(v31.*(p1+p3)*.5,2);

det23=(v21(:,2).*v31(:,3))-(v21(:,3).*v31(:,2));
det13=(v21(:,3).*v31(:,1))-(v21(:,1).*v31(:,3));
det12=(v21(:,1).*v31(:,2))-(v21(:,2).*v31(:,1));

Det=v41(:,1).*det23+v41(:,2).*det13+v41(:,3).*det12;


detx=d1.*det23+...
    v41(:,2).*(-(d2.*v31(:,3))+(v21(:,3).*d3))+...
    v41(:,3).*((d2.*v31(:,2))-(v21(:,2).*d3));

dety=v41(:,1).*((d2.*v31(:,3))-(v21(:,3).*d3))+...
    d1.*det13+...
    v41(:,3).*((d3.*v21(:,1))-(v31(:,1).*d2));

detz=v41(:,1).*((v21(:,2).*d3)-(d2.*v31(:,2)))...
    +v41(:,2).*(d2.*v31(:,1)-v21(:,1).*d3)...
    +d1.*(det12);

%Circumcenters
cc(:,1)=detx./Det;
cc(:,2)=dety./Det;
cc(:,3)=detz./Det;


%Circumradius
r=realsqrt((sum((p2-cc).^2,2)));%squared radius
end




%% Intersection factor
function Ifact=IntersectionFactor(tetr2t,cc,r)
nt=size(tetr2t,1);
Ifact=zeros(nt,1);%intersection factor
%it is computed from radius of the tetraedroms circumscribed sphere
% and the distance between their center
i=tetr2t(:,2)>0;
 distcc=sum((cc(tetr2t(i,1),:)-cc(tetr2t(i,2),:)).^2,2);%distance between circumcenters
  Ifact(i)=(-distcc+r(tetr2t(i,1)).^2+r(tetr2t(i,2)).^2)./(2*r(tetr2t(i,1)).*r(tetr2t(i,2)));

% for i=1:nt
%     if tetr2t(i,2)>0 %jump boundary tetraedrom
%         distcc=sum((cc(tetr2t(i,1),:)-cc(tetr2t(i,2),:)).^2,2);%distance between circumcenters
%         %intersection factor
%         Ifact(i)=(-distcc+r(tetr2t(i,1))^2+r(tetr2t(i,2))^2)/(2*r(tetr2t(i,1))*r(tetr2t(i,2)));
%     end
% end
end







%% AddShield
function [pnew,nshield]=AddShield(p)

%adds outside points to the given cloud forming outside tetraedroms

%shield points are very good in detectinf outside tetraedroms. Unfortunatly
%delunany triangulation with these points can be even of 50% slower.

%find the bounding box
maxx=max(p(:,1));
maxy=max(p(:,2));
maxz=max(p(:,3));
minx=min(p(:,1));
miny=min(p(:,2));
minz=min(p(:,3));

%give offset to the bounding box
% step=max(abs([maxx-minx,maxy-miny,maxz-minz]));

% step=max(abs([maxx,minx,maxy,miny,maxz,minz]));

offset=5;
stepx=(maxx-minx)*offset;
stepy=(maxy-miny)*offset;
stepz=(maxz-minz)*offset;

maxx=maxx+stepx;
maxy=maxy+stepy;
maxz=maxz+stepz;
minx=minx-stepx;
miny=miny-stepy;
minz=minz-stepz;

N=4;%number of points of the shield edge

stepx=stepx/(N);%decrease step, avoids not unique points
stepz=stepz/(N);%decrease step, avoids not unique points

nshield=N*N*6;

%creating a grid lying on the bounding box
vx=linspace(minx,maxx,N);
vy=linspace(miny,maxy,N);
vz=linspace(minz,maxz,N);




[x,y]=meshgrid(vx,vy);
facez1=[x(:),y(:),ones(N*N,1)*maxz];
facez2=[x(:),y(:),ones(N*N,1)*minz];
[x,y]=meshgrid(vy,vz-stepz);
facex1=[ones(N*N,1)*maxx,x(:),y(:)];
facex2=[ones(N*N,1)*minx,x(:),y(:)];
[x,y]=meshgrid(vx-stepx,vz);
facey1=[x(:),ones(N*N,1)*maxy,y(:)];
facey2=[x(:),ones(N*N,1)*miny,y(:)];

%add points to the p array
pnew=[p;
    facex1;
    facex2;
    facey1;
    facey2;
    facez1;
    facez2];

% figure(5)
% plot3(pnew(:,1),pnew(:,2),pnew(:,3),'.g')

end


%% Manifold Extraction

function t=ManifoldExtraction(t,p)
%Given a set of trianlges,
%Buils a manifolds surface with the ball pivoting method.



% building the etmap

numt = size(t,1);
vect = 1:numt;                                                             % Triangle indices
e = [t(:,[1,2]); t(:,[2,3]); t(:,[3,1])];                                  % Edges - not unique
[e,j,j] = unique(sort(e,2),'rows');                                        % Unique edges
te = [j(vect), j(vect+numt), j(vect+2*numt)];
nume = size(e,1);
e2t  = zeros(nume,2,'int32');

clear vect j
ne=size(e,1);
np=size(p,1);


count=zeros(ne,1,'int32');%numero di triangoli candidati per edge
etmapc=zeros(ne,4,'int32');
for i=1:numt

    i1=te(i,1);
    i2=te(i,2);
    i3=te(i,3);



    etmapc(i1,1+count(i1))=i;
    etmapc(i2,1+count(i2))=i;
    etmapc(i3,1+count(i3))=i;


    count(i1)=count(i1)+1;
    count(i2)=count(i2)+1;
    count(i3)=count(i3)+1;
end

etmap=cell(ne,1);
for i=1:ne

    etmap{i,1}=etmapc(i,1:count(i));

end
clear  etmapc

tkeep=false(numt,1);%all'inizio nessun trinagolo selezionato


%Start the front

%building the queue to store edges on front that need to be studied
efront=zeros(nume,1,'int32');%exstimate length of the queue

%Intilize the front


         tnorm=Tnorm(p,t);%get traingles normals
         
         %find the highest triangle
         [foo,t1]=max( (p(t(:,1),3)+p(t(:,2),3)+p(t(:,3),3))/3);

         if tnorm(t1,3)<0
             tnorm(t1,:)=-tnorm(t1,:);%punta verso l'alto
         end
         
         %aggiungere il ray tracing per verificare se il triangolo punta
         %veramente in alto.
         %Gli altri triangoli possono essere trovati sapendo che se un
         %triangolo ha il baricentro più alto sicuramente contiene il punto
         %più alto. Vanno analizzati tutto i traingoli contenenti questo
         %punto
         
         
            tkeep(t1)=true;%primo triangolo selezionato
            efront(1:3)=te(t1,1:3);
            e2t(te(t1,1:3),1)=t1;
            nf=3;%efront iterato
      

while nf>0


    k=efront(nf);%id edge on front

    if e2t(k,2)>0 || e2t(k,1)<1 || count(k)<2 %edge is no more on front or it has no candidates triangles

        nf=nf-1;
        continue %skip
    end
  
   
      %candidate triangles
    idtcandidate=etmap{k,1};

    
     t1=e2t(k,1);%triangle we come from
    
   
        
    %get data structure
%        p1
%       / | \
%  t1 p3  e1  p4 t2(idt)
%       \ | /  
%        p2
         alphamin=inf;%inizilizza
          ttemp=t(t1,:);
                etemp=e(k,:);
                p1=etemp(1);
                p2=etemp(2);
                p3=ttemp(ttemp~=p1 & ttemp~=p2);%terzo id punto
        
                
         %plot for debug purpose
%          close all
%          figure(1)
%          axis equal
%          hold on
%          
%          fs=100;
%         
%          cc1=(p(t(t1,1),:)+p(t(t1,2),:)+p(t(t1,3),:))/3;
%          
%          trisurf(t(t1,:),p(:,1),p(:,2),p(:,3))
%          quiver3(cc1(1),cc1(2),cc1(3),tnorm(t1,1)/fs,tnorm(t1,2)/fs,tnorm(t1,3)/fs,'b');
%                 
       for i=1:length(idtcandidate)
               t2=idtcandidate(i);
               if t2==t1;continue;end;
                
               %debug
%                cc2=(p(t(t2,1),:)+p(t(t2,2),:)+p(t(t2,3),:))/3;
%          
%                 trisurf(t(t2,:),p(:,1),p(:,2),p(:,3))
%                 quiver3(cc2(1),cc2(2),cc2(3),tnorm(t2,1)/fs,tnorm(t2,2)/fs,tnorm(t2,3)/fs,'r');
%                
%                

               
                ttemp=t(t2,:);
                p4=ttemp(ttemp~=p1 & ttemp~=p2);%terzo id punto
        
   
                %calcola l'angolo fra i triangoli e prendi il minimo
              
                
                [alpha,tnorm2]=TriAngle2(p(p1,:),p(p2,:),p(p3,:),p(p4,:),tnorm(t1,:));
                
                if alpha<alphamin
                    
                    alphamin=alpha;
                    idt=t2;  
                    tnorm(t2,:)=tnorm2;%ripristina orientazione   
                     
                    %debug
%                      quiver3(cc2(1),cc2(2),cc2(3),tnorm(t2,1)/fs,tnorm(t2,2)/fs,tnorm(t2,3)/fs,'c');
                    
                end
                %in futuro considerare di scartare i trianoli con angoli troppi bassi che
                %possono essere degeneri
                
       end


   
   
    
    
   %update front according to idttriangle
          tkeep(idt)=true;
        for j=1:3
            ide=te(idt,j);
           
            if e2t(ide,1)<1% %Is it the first triangle for the current edge?
                efront(nf)=ide;
                nf=nf+1;
                e2t(ide,1)=idt;
            else                     %no, it is the second one
                efront(nf)=ide;
                nf=nf+1;
                e2t(ide,2)=idt;
            end
        end
        
     
        

         nf=nf-1;%per evitare di scappare avanti nella coda e trovare uno zero
end
t=t(tkeep,:);
end





%% TriAngle2
function  [alpha,tnorm2]=TriAngle2(p1,p2,p3,p4,planenorm)

%per prima cosa vediamo se il p4 sta sopra o sotto il piano identificato
%dalla normale planenorm e il punto p3

test=sum(planenorm.*p4-planenorm.*p3);



%Computes angle between two triangles
v21=p1-p2;
v31=p3-p1;

tnorm1(1)=v21(2)*v31(3)-v21(3)*v31(2);%normali ai triangoli
tnorm1(2)=v21(3)*v31(1)-v21(1)*v31(3);
tnorm1(3)=v21(1)*v31(2)-v21(2)*v31(1);
tnorm1=tnorm1./norm(tnorm1);



v41=p4-p1;
tnorm2(1)=v21(2)*v41(3)-v21(3)*v41(2);%normali ai triangoli
tnorm2(2)=v21(3)*v41(1)-v21(1)*v41(3);
tnorm2(3)=v21(1)*v41(2)-v21(2)*v41(1);
tnorm2=tnorm2./norm(tnorm2);
alpha=tnorm1*tnorm2';%coseno dell'angolo
%il coseno considera l'angolo fra i sempipiani e non i traigoli, ci dice
%che i piani sono a 180 se alpha=-1 sono concordi se alpha=1, a 90°

alpha=acos(alpha);%trova l'angolo

%Se p4 sta sopra il piano l'angolo è quello giusto altrimenti va maggiorato
%di 2*(180-alpha);

if test<0%p4 sta sotto maggioriamo
   alpha=alpha+2*(pi-alpha);
end

%         fs=100;
%          cc2=(p1+p2+p3)/3;
%        quiver3(cc2(1),cc2(2),cc2(3),tnorm1(1)/fs,tnorm1(2)/fs,tnorm1(3)/fs,'m');
%        cc2=(p1+p2+p4)/3;
%               quiver3(cc2(1),cc2(2),cc2(3),tnorm2(1)/fs,tnorm2(2)/fs,tnorm2(3)/fs,'m');

%vediamo se dobbiamo cambiare l'orientazione del secondo triangolo
%per come le abbiamo calcolate ora tnorm1 t tnorm2 non rispettano
%l'orientamento
testor=sum(planenorm.*tnorm1);
if testor>0 
    tnorm2=-tnorm2;
end



end


%% Tnorm

function tnorm1=Tnorm(p,t)
%Computes normalized normals of triangles


v21=p(t(:,1),:)-p(t(:,2),:);
v31=p(t(:,3),:)-p(t(:,1),:);

tnorm1(:,1)=v21(:,2).*v31(:,3)-v21(:,3).*v31(:,2);%normali ai triangoli
tnorm1(:,2)=v21(:,3).*v31(:,1)-v21(:,1).*v31(:,3);
tnorm1(:,3)=v21(:,1).*v31(:,2)-v21(:,2).*v31(:,1);

L=sqrt(sum(tnorm1.^2,2));

tnorm1(:,1)=tnorm1(:,1)./L;
tnorm1(:,2)=tnorm1(:,2)./L;
tnorm1(:,3)=tnorm1(:,3)./L;
end
