function [edges,N,coords] = lattice3(X,Y,T,Opt)
%@JK: Modify to be more general regular connected graph
%Opt: '4Connect', '8Connect', 'Diag8Connect'

% [edges,N,coords] = lattice3(X,Y,T) 
% edges is an Mx2 array, where each row corresponds to a pair of vertices
% which are connected.
%
% N = X*Y*T is the number of nodes, i.e., the resolution of the 
% image (in total pixels per frame) times the number of frames.
%
% coords is an Nx3 array, where each row is a Euclidean coordinate for 
% a node in the lattice, which can, e.g., be used for visualization. 
% See latticedemo.m for an example.
%
%
% Note, the number of edges E of a 3D rectangular lattice whose dimensions
% are given by X,Y,T is:
%
% E = (X-1)*Y*T + X*(Y-1)*T + X*Y*(T-1)
%
% For a 2D image, T=1, and this reduces to E=(X-1)*Y + X*(Y-1).
%
% For example, with the input:
%
% [edges,N,points] = lattice3(100,200,50)
%
% We have:
% E = (100-1)*200*50 + 100*(200-1)*50 + 100*200*(50-1) = 2 965 000
% N = 100*200*50 = 1 000 000
%
% and the output should be:
%
% edges, an array of size 2 965 000-by-2
% N=1,000,000
% coords, an array of size 1 000 000-by-3
%

% License: BSD
% Copyright 2011 Manuchehr Aminian, Andrew V. Knyazev

% Input checking.
if nargin==1
    T=1; Y=1; Opt='4Connect';
elseif nargin==2
    T=1; Opt='4Connect';
elseif nargin==3
    Opt='4Connect';
end


% Calculate number of points.
N=X*Y*T;

% Generate coordinate triplets of points, if desired.
if nargout == 3
    coords=zeros(N,3);
    
    % f is used to construct the coordinate triplets. When the points are 
    % ordered regularly, the patterns in the coords can be described in 
    % this way.  The function generates a vector that cycles through the 
    % integers 0...(n-1) c times, repeating each digit k times.
    f=@(n,k,c) mod(ceil((1:(n*k*c))'/k)-1,n);
    
    coords(:,1)=f(X,Y,T);
    coords(:,2)=f(Y,1,X*T);
    coords(:,3)=f(T,X*Y,1);
end

% Create edge connections.  Note, temp is simply a placeholder so that the
% lines here look a little nicer.  The formula here should describe the
% patterns that show up when the vertices are numbered regularly.
% Try playing around with latticedemo.m to convince yourself this works.

if ( strcmp(Opt , '4Connect') )
% formula for total number of edges.
E=(X-1)*Y*T + X*(Y-1)*T + X*Y*(T-1);
edges=zeros(E,2);

% Use a 'pointer' to keep track of the current index location in the 
% edges array.
loc=0;

% Construct edges in the Y-direction
temp=(1:(Y-1)*X*T)';
edges(loc+1:loc+(Y-1)*X*T,1:2)=[temp,temp+1]+floor([temp-1,temp-1]/(Y-1));

% Shift the pointer.
loc=loc+(Y-1)*X*T;
  
% Construct edges in the X-direction
temp=(1:Y*(X-1)*T)';
edges(loc+1:loc+Y*(X-1)*T,1:2)=...
    [temp,temp] + [zeros(size(temp)),Y*ones(size(temp))] + ...
    Y*floor([temp-1,temp-1]/((X-1)*Y));

% Shift the pointer again.
loc=loc+Y*(X-1)*T;

% Connect the first frame to every other frame
temp=repmat(1:X*Y, 1, T-1)';
temp2=(X*Y+1:X*Y*T)';
edges(loc+1:E,1:2)=[temp,temp2];

elseif ( strcmp(Opt , '8Connect') )
    % total number of edges
    E = (X-1)*Y*T + X*(Y-1)*T + X*Y*(T-1) + (X-2)*Y*T + X*(Y-2)*T + X*Y*(T-2);
    edges=zeros(E,2);
    loc=0;
    % Construct 1-hop edges in the Y-direction
    temp=(1:(Y-1)*X*T)';
    edges(loc+1:loc+(Y-1)*X*T,1:2)=[temp,temp+1]+floor([temp-1,temp-1]/(Y-1));
    loc=loc+(Y-1)*X*T;
    
    % Construct 1-hop edges in the X-direction
    temp=(1:Y*(X-1)*T)';
    edges(loc+1:loc+Y*(X-1)*T,1:2)=...
        [temp,temp] + [zeros(size(temp)),Y*ones(size(temp))] + ...
        Y*floor([temp-1,temp-1]/((X-1)*Y));
    loc=loc+Y*(X-1)*T;
    
    % Construct 2-hop edges in the Y-direction
    temp=(1:(Y-2)*X*T)';
    edges(loc+1:loc+(Y-2)*X*T,1:2)=[temp,temp+2]+2.*floor([temp-1,temp-1]/(Y-2));
    loc=loc+(Y-2)*X*T;
    
    % Construct 2-hop edges in the X-direction
    temp=(1:Y*(X-2)*T)';
    edges(loc+1:loc+Y*(X-2)*T,1:2)=...
        [temp,temp] + [zeros(size(temp)),2*Y*ones(size(temp))];
    loc=loc+Y*(X-2)*T;
    
elseif ( strcmp(Opt , 'Diag8Connect') )
    % total number of edges
    E = (X-1)*Y*T + X*(Y-1)*T + X*Y*(T-1) + (X-1)*(Y-1)*T*2;
    edges=zeros(E,2);
    loc=0;
    % Construct 1-hop edges in the Y-direction
    temp=(1:(Y-1)*X*T)';
    edges(loc+1:loc+(Y-1)*X*T,1:2)=[temp,temp+1]+floor([temp-1,temp-1]/(Y-1));
    loc=loc+(Y-1)*X*T;
    
    % Construct 1-hop edges in the X-direction
    temp=(1:Y*(X-1)*T)';
    edges(loc+1:loc+Y*(X-1)*T,1:2)=...
        [temp,temp] + [zeros(size(temp)),Y*ones(size(temp))] + ...
        Y*floor([temp-1,temp-1]/((X-1)*Y));
    loc=loc+Y*(X-1)*T;
    
    % Construct diagonal edges in the top-left to bot-right direction
    temp = (1:(Y-1)*(X-1)*T)';
    edges(loc+1:loc+(Y-1)*(X-1)*T,1:2)=... 
        [temp,temp+1]+floor([temp-1,temp-1]/(Y-1))+[zeros(size(temp)),Y*ones(size(temp))];
    loc=loc+(Y-1)*(X-1)*T;

    % Construct diagonal edges in the bot-left to top-right direction
    temp = (1:(Y-1)*(X-1)*T)';
    edges(loc+1:loc+(Y-1)*(X-1)*T,1:2)=...
        [temp+1,temp]+floor([temp-1,temp-1]/(Y-1))+[zeros(size(temp)),Y*ones(size(temp))];
    loc=loc+(Y-1)*(X-1)*T;
    
end

end
