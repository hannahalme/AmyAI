import numpy as np
import pydicom
import json
import argparse
from pathlib import Path
import requests
import os
import sys
import csv


def parse_args():
    p = argparse.ArgumentParser(description="Batch predict using TF Serving REST API.")
    p.add_argument(
        "--input-dir",
        default=os.environ.get("INPUT_DIR", "/data"),
        help="Directory containing input files (default: env INPUT_DIR or /data).",
    )
    p.add_argument(
        "--pattern",
        default="*.dcm",
        help="Glob pattern to select input files (default: *.dcm).",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "/out"),
        help="Directory to write predictions (default: env OUTPUT_DIR or /out).",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("TF_HOST", "tfserving"),
        help="TF Serving host (default: env TF_HOST or tfserving).",
    )
    p.add_argument(
        "--port",
        default=os.environ.get("TF_PORT", "8501"),
        help="TF Serving REST port (default: env TF_PORT or 8501).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", "ResNet2"),
        help="Model name (default: env MODEL_NAME or ResNet2).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout seconds (default: 30).",
    )
    return p.parse_args()



def crop_WB(img):

    # Find the y position of shoulders
    img_mean = np.mean(img)
    for i in range(img.shape[0]):
        profilex = img[i,:]
        # Find pixels with activity
        active_ids = np.where(profilex > img_mean)
        count = np.shape(active_ids)[1]
        # Check that there are >100 consecutive pixels
        if count > 100:
            diffs = np.ediff1d(active_ids)
            if np.max(diffs)<10:
                y0 = i
                break
                
    # 128 pixels in y-direction
    y1 = int(y0 + 128)
    
    # Find the center of the patient
    profilex_m = np.mean(img[y0:y1, :], axis=0)
    ids = np.where(profilex_m > img_mean)
    center = np.mean([ids[0], ids[-1]])
    
    # Make 128 pixels wide cropped matrix, centered on patient
    x0 = int(center - 64)
    x1 = int(x0 + 128)
    cropped = img[y0:y1,x0:x1]

    # Log transform
    cropped = np.log(cropped + 1e-8)
    print("Image cropped!")

    return cropped
    
    
def main():

    args = parse_args()
    
    # Dicom tags to be extracted
    tags = ['AccessionNumber','SOPInstanceUID','StudyDate','StudyTime','StationName', 'PatientSex','PatientAge']

    # Initialize data dictionary
    data_dict = {tag: [] for tag in tags}
    data_dict['prediction'] = []
    data_dict['probability'] = []
    
    # Training data mean and std
    if args.model == "ResNet2":
        train_mean = 3.3898125
        train_std = 0.60752076
    elif args.model == "ResNet2_2026":
        train_mean = 3.306355
        train_std = 0.770120
    else:
        print("No model named ", args.model)
        return
        

    # Loop over all files in root directory
    for root, _, files  in os.walk(args.input_dir):

	    for i in files:
	    
		    try:
			    ds = pydicom.dcmread(os.path.join(root,i))
			    imgtype = ds.ImageType
		    except:
			    print("Cannot read DICOM image or ImageType unknown")
			    continue
		    
		    # Process only whole body images
		    if "WHOLE BODY" in imgtype:
		    
		        # Discard print screen images
			    if "SECONDARY" in imgtype:
				    print("Secondary capture, not processed!")
				    continue
			        
			    # Get image matrix
			    try:
				    img = ds.pixel_array
			    except:
				    print("No DICOM image!")
				    continue
			    
			    # Discard PA images 
			    if(len(img)==0 or len(img.shape)<2):
				    print("No image matrix!")
				    continue
			    elif len(img.shape)==2:
				    if hasattr(ds, "ImageID"):
					    if (ds.ImageID == "POSTERIOR"):
						    print("PA image not processed!")
						    continue
			    elif (img.shape[2]) == 3:
				    print("Print screen, not processed!")
				    continue
			    else:
				    img = img[0]

			    # Crop whole body image
			    cropped = crop_WB(img)


			    # Normalize current image with respect to training mean and std
			    cropped = (cropped - train_mean) / train_std

			    
			    # Reshape the cropped image to tensorflow shape
			    cropped = cropped.reshape(1, cropped.shape[0], cropped.shape[1], 1)
			    

			    
			    # Make a JSON file of data: image to list
			    data = json.dumps({"signature_name": "serving_default", "instances": cropped.tolist()})
			    headers = {"content-type": "application/json"}
			    
			    # Post a response to tfserving
			    url = f"http://{args.host}:{args.port}/v1/models/{args.model}:predict"
			    response = requests.post(url, data=data, headers=headers)
			    result = response.json()["predictions"][0]
			    

			    # Print prediction
			    pred1d = int(np.argmax(result))
			    classes = ["G0-1","G2-3"]
			    print("Myocardial uptake classification is with", np.max(result)*100, "% probability", classes[pred1d])
			    
		        

			    # Add study information to data dictionary	
			    for tag in tags:
			        try:
			            data_dict[tag].append(str(getattr(ds, tag)))
			        except:
			            data_dict[tag].append(None)
			    data_dict['prediction'].append(pred1d)
			    data_dict['probability'].append(np.max(result))
			    

		    elif "AXIAL" in imgtype:
			    print("CT image")
			    break
		    
		    else:
			    print("Not a WB image!")
			    continue


    # Write dictionary to json file		
    json.dump(data_dict, open(args.output_dir + "/test_data_classification_results.json",'w'))
    
    # Write csv file
    with open(args.output_dir + "/test_data_classification_results.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(data_dict.keys())
        writer.writerows(zip(*data_dict.values()))
    

if __name__ == "__main__":
    main()

