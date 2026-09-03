#!/usr/bin/env python3
"""Audit image splits for exact duplicates and close perceptual-hash matches."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageFile
from scipy.fft import dctn
ImageFile.LOAD_TRUNCATED_IMAGES=True
POPCOUNT=np.array([bin(v).count('1') for v in range(256)],dtype=np.uint8)
def bits_to_int(bits):
    value=0
    for bit in bits.ravel(): value=(value<<1)|int(bit)
    return value
def phash(image,crop=False):
    if crop:
        width,height=image.size; image=image.crop((int(width*.08),int(height*.08),int(width*.92),int(height*.92)))
    values=dctn(np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32),norm='ortho')[:8,:8].ravel(); bits=values>np.median(values[1:]); bits[0]=False; return bits_to_int(bits)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def collect(root,split):
    rows=[]
    for path in sorted(root.glob('*/*')):
        if path.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
        with Image.open(path) as image: image.load(); rows.append({'split':split,'class':path.parent.name,'path':str(path),'sha256':digest(path),'phash':phash(image),'crop_phash':phash(image,True)})
    return rows
def hamming(block,refs): return POPCOUNT[np.bitwise_xor(block[:,None],refs[None,:]).view(np.uint8)].reshape(len(block),len(refs),8).sum(axis=2)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--development-root',type=Path,required=True); parser.add_argument('--held-out-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); development=collect(args.development_root,'development'); held=collect(args.held_out_root,'held_out'); lookup=defaultdict(list)
    for i,row in enumerate(development): lookup[row['sha256']].append(i)
    exact=sum(len(lookup[row['sha256']]) for row in held); dev_full=np.array([r['phash'] for r in development],dtype=np.uint64); dev_crop=np.array([r['crop_phash'] for r in development],dtype=np.uint64); near4=near6=0
    for start in range(0,len(held),64):
        block=held[start:start+64]; full=hamming(np.array([r['phash'] for r in block],dtype=np.uint64),dev_full); crop=hamming(np.array([r['crop_phash'] for r in block],dtype=np.uint64),dev_crop); nearest=np.argmin(full.astype(np.uint16)+crop.astype(np.uint16),axis=1)
        for offset,index in enumerate(nearest): near4+=int(full[offset,index]<=4 and crop[offset,index]<=4); near6+=int(full[offset,index]<=6 and crop[offset,index]<=6)
    result={'development_images':len(development),'held_out_images':len(held),'total_images':len(development)+len(held),'exact_cross_split_duplicates':exact,'near_pair_threshold_counts':{'both_phash_le_4':near4,'both_phash_le_6':near6},'development_counts_by_class':Counter(r['class'] for r in development),'held_out_counts_by_class':Counter(r['class'] for r in held)}; args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
