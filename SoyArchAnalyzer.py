import os, glob
import sys
import pandas as pd
from datetime import date
import argparse
import numpy as np
import cv2 as cv
import imutils
import pandas as pd
import time
from skimage.morphology import skeletonize
from fil_finder import FilFinder2D
import astropy.units as u
from transformers import Sam3Processor, Sam3Model
import torch
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

date = date.today().strftime("%Y%m%d")
start_time = time.perf_counter()

        
def main():
    # Initialize options
    # args = options()
    # folder,save= input("enter foldername directory and output directory: ").split()
    # input_folder=os.path.abspath(folder)
    # output_folder=os.path.abspath(save)

    result = f'SoyArchAnalysisResults_{date}.csv'

    parser = argparse.ArgumentParser(description="Enter input and output directory.")
    parser.add_argument("indir", type=str, help="Input directory with no final /")
    parser.add_argument("outdir", type=str, help="Output directory with no final /")
    args = parser.parse_args()
    input_folder = os.path.abspath(args.indir)
    output_folder= os.path.abspath(args.outdir)
    os.makedirs(output_folder, exist_ok=True)



    ext="jpeg"
    files = glob.glob(os.path.join(input_folder, f'*.{ext}'))
    
    df = pd.DataFrame()

    for im in files:
        
        image_name = im.replace(f'{input_folder}/', "").replace(".jpg","")

        print(f"Processing: {image_name}")
        start1 = time.perf_counter()
        
        #args.image = input_folder+i
        args.image = im
        
        #Read in image
        im = cv.imread(args.image)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = im[400:2800, 400:2230]


        # Segment using text prompt
        inputs = processor(images=[im,im], text=["leaf", "stem"], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.62,
            mask_threshold=0.62,
            target_sizes=inputs.get("original_sizes").tolist()
        )


        #First, stems
        stemmasks = results[1]["masks"]
        #filter out incorrect masks
        filtered_mask_ids = []
        for idx, mask in enumerate(stemmasks):
            if mask[0:1500, 1400:2000].sum()==0:
                filtered_mask_ids.append(idx)
        stemmasks= stemmasks[filtered_mask_ids]
        stemboxes = results[1]["boxes"][filtered_mask_ids]

        stem_df = []
        out_im = im.copy()
        for idx,stem in enumerate(stemmasks):
            stem = np.array(stem, np.uint8)
            if stem.sum()>10000:
                skeleton = skeletonize(stem)
                fil = FilFinder2D(stem, mask=skeleton)
                fil.medskel(verbose=False)
                fil.analyze_skeletons(skel_thresh=50*u.pix, branch_thresh=10*u.pix,
                                      prune_criteria='length', max_prune_iter=1)
                skel = np.asarray(fil.skeleton_longpath).squeeze()
                length = round(np.count_nonzero(skel)/53,2)
                
                contours, _ = cv.findContours(
                np.array(fil.skeleton_longpath, np.uint8),
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_NONE
                )

                cv.drawContours(out_im, contours[0], -1, (255,0,0), 5)
                
                #Draw line analyzed
                skel = fil.skeleton_longpath.squeeze().astype(np.uint8)
                ys, xs = np.nonzero(skel)
                cx = int(xs.mean())
                cy = int(ys.mean())

                cv.putText(out_im, str(length), (cx,cy), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
                
            else:
                stem = np.array(stem, np.uint8)
                [x0,y0,x1,y1] = np.array(stemboxes[idx]).astype(int)
                h = y1-y0
                length = round(h/53,2)
                
         
                contours, _ = cv.findContours(
                    stem,
                    cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_NONE
                )
                cont = max(contours, key=cv.contourArea)
                w = x1-x0
                cv.line(out_im, (x0+(w//2),y0),(x0+(w//2),y1), (255,0,255), 6)
                cv.putText(out_im, str(length), (x0+(w//2), y0+(h//2)), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)

            
                #Save length
            stem_df.append({
                'Object': f"stem_{idx + 1}",    
                'Trait': "Length(cm)",
                'Value': length
                
            })


            #Now leaves
            leafmasks = results[0]["masks"]
            #filter out incorrect masks
            filtered_mask_ids = []
            for idx, mask in enumerate(leafmasks):
                if np.array(mask.sum() >=10000):
                    filtered_mask_ids.append(idx)
                
            leafmasks= leafmasks[filtered_mask_ids]
            result_boxes = results[0]["boxes"][filtered_mask_ids]


            leaf_df = []
            for idx,leaf in enumerate(leafmasks):
                leaf = np.array(leaf, np.uint8)*255
                #Get height and width
                [x0,y0,x1,y1] = np.array(result_boxes[idx]).astype(int)
                m1 = x1-x0
                m2 = y1-y0
                
                if m1<m2:
                    width, height = m1,m2
                else:
                    width, height = m2,m1
                
                #Get area
                contours, _ = cv.findContours(
                    np.array(leaf, np.uint8),
                    cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_NONE
                )
                
                leaf_area = cv.contourArea(contours[0])
                
                #Save traits
                leaf_df.append({
                    'Object': f"leaflet_{idx + 1}",    
                    'Area(cm2)': round(leaf_area/(53*53),2),
                    'Height(cm)': round(height/53,2),
                    'Width(cm)': round(width/53,2)
                    
                })
                
                #Draw measurements
                leaf = cv.cvtColor(leaf, cv.COLOR_GRAY2BGR)
                cv.drawContours(out_im, contours, -1, (0,255,0), 6)
                cv.line(out_im, (x0+(m1//2), y0), (x0+(m1//2),y0+m2), (255,0,255), 6)
                cv.putText(out_im, str(round(m2/53,2)), (x0+(m1//2),y0+m2), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
                cv.line(out_im, (x0, y0+(m2//2)), (x1,y0+(m2//2)), (255,0,255), 6)
                cv.putText(out_im, str(round(m1/53,2)), (x1,y0+(m2//2)), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)


            #Merge datasets
            df_unit = pd.concat([
            pd.DataFrame(stem_df), 
            pd.melt(pd.DataFrame(leaf_df), 
                    id_vars=['Object'], 
                    var_name='Trait', 
                    value_name='Value')
            ])

        output_path = os.path.join(output_folder, f'processed_{image_name}.jpg')
        masked_image = cv.cvtColor(out_im, cv.COLOR_BGR2RGB)
        cv.imwrite(output_path, masked_image)    

        
        df_unit.insert(0, 'image', image_name)

        df = pd.concat([df, df_unit], ignore_index=True)
        
        end1 = time.perf_counter()
        elapsed1 = end1 - start1
        print(f"Done processing {image_name} - took {elapsed1:.4f} seconds")



    # Save to CSV in output folder
    df_output_path = os.path.join(output_folder, result)
    df.to_csv(df_output_path, index=False)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Done running all images in: {elapsed_time:.4f} seconds")

        
if __name__ == "__main__":
    main() 