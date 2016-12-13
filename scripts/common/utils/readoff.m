function [node,elem]=readoff(fname)
% [node,elem]=readoff(fname)
%
% readoff: read  Geomview Object File Format
%
% author: fangq (fangq<at> nmr.mgh.harvard.edu)
% date: 2008/03/28
%
% input:
%    fname: name of the OFF data file
% output:
%    node: node coordinates of the mesh
%    elem: list of elements of the mesh	    
%
% -- this function is part of iso2mesh toolbox (http://iso2mesh.sf.net)
%

node=[];
elem=[];
fid=fopen(fname,'rt');

line=fgetl(fid); %off

flag = 1;
if ~isempty(line)
    if str2num(line(1))
       flag = 0; 
    end
else
   flag = 1; 
end

while flag
    line=fgetl(fid);
    
    if ~isempty(line)
        if str2num(line(1))
           flag = 0; 
           dim=sscanf(line,'%d',3);
        end
    end
end
    
%dim=fscanf(fid,'%d',3);
node=fscanf(fid,'%f',[3,dim(1)])';
elem=fscanf(fid,'%f',inf);
if(length(elem)==4*dim(2))
    elem=reshape(elem,[4,dim(2)])';
elseif(length(elem)==8*dim(2))
    elem=reshape(elem,[8,dim(2)])';
end
if(size(elem,2)<=3)
    elem=round(elem(:,2:3))+1;
else
    elem=round(elem(:,2:4))+1;
end

fclose(fid);