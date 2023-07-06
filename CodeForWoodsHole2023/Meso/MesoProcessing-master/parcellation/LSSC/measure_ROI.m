function out_list=measure_ROI(im,in_list)
    nROI=length(in_list);
    ss=size(im);
    out_list=[];
    for i=1:nROI
        tempROI.pixel_list=in_list(i).pixel_list;
        ROImap=false(ss(1:2));
        ROImap(in_list(i).pixel_list)=1;
        
        if isfield(in_list(i),'boundary_list')&& (~isempty(in_list(i).boundary_list))
            tempROI.boundary_list=in_list(i).boundary_list;
        else
            tempROI.boundary_list=find(bwperim(ROImap,4));
        end
        
        
        [I1,J1]=ind2sub(ss,tempROI.pixel_list);
        tempROI.centerPos=[mean(I1),mean(J1)];
        
        
        if isfield(in_list(i),'seedPos')&& (~isempty(in_list(i).seedPos))
            tempROI.seedPos=in_list(i).seedPos;
        else
            tempROI.seedPos=tempROI.centerPos;
        end
        
        if isfield(in_list(i),'name')&& (~isempty(in_list(i).name))
            tempROI.name=in_list(i).name;
        else
            tempROI.name='ROI';
        end
        
        ROImap_fill=imfill(ROImap,'holes');
        im=reshape(im,[],ss(3));
        tempROI.fmean=mean(im(tempROI.pixel_list,:))';
        
        tempROI.fill_list=find(ROImap_fill);
        tempROI.fmean_fill=mean(im(tempROI.fill_list,:))';

        im=reshape(im,ss);
        out_list=[out_list,tempROI];
    end   
end