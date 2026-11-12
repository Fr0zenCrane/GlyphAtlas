"""Portable parent-level annotation selection; no training imports."""
import hashlib

def select_crop(parent,epoch):
    variants=parent['crop_variants']
    if parent.get('parent_sampling_weight')!=1 or not 1<=len(variants)<=2:
        raise ValueError('Expected one parent with one or two crops')
    offset=int.from_bytes(hashlib.sha256(parent['key'].encode()).digest()[:8],'big')
    selected=variants[(offset+int(epoch))%len(variants)]
    if selected['annotation']['img_name']!=selected['image']:
        raise ValueError('Crop image and annotation identity disagree')
    return selected

def select_record(parent,index,epoch,default_image):
    """Image and annotation must be selected together, never independently."""
    if 'crop_variants' in parent:
        crop=select_crop(parent,epoch)
        return crop['annotation'],crop['image'],True
    return select_annotation(parent,index,epoch),default_image,False

def select_annotation(parent,index,epoch):
    if 'variants' not in parent:return parent
    variants=parent['variants']
    if not variants:raise ValueError('Parent has no annotation variants')
    if parent.get('parent_sampling_weight')!=1:raise ValueError('Parent sampling weight must be one')
    offset=int.from_bytes(hashlib.sha256(parent['key'].encode()).digest()[:8],'big')
    return variants[(offset+int(epoch))%len(variants)]['annotation']
