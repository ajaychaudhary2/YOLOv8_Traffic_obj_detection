from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO


app= Flask(__name__)

BASE_DIR = r"E:\Data_Science _master\DL\CNN\Object detection\Yolo8\traffic_detection_webapp_yolov8"
UPLOAD_F=os.path.join(BASE_DIR,"static","uploads")
PROCCESSED_F=os.path.join(BASE_DIR,"static","predictions")

app.config["UPLOAD_F"]=UPLOAD_F
app.config["PROCCESSED_F"]=PROCCESSED_F


os.makedirs(UPLOAD_F,exist_ok=True)
os.makedirs(PROCCESSED_F,exist_ok=True)

 #loading the model
model_path = os.path.join(BASE_DIR,"yolov8m.pt")
model=YOLO(model_path)

@app.route("/",methods=["GET","POST"])

def upload_img():
    if request.method=="POST":
        if "file" not in request.files:
            return(redirect.url)
        
        file=request.files["file"]
        if "file" == "":
            return(redirect.url)
        
        if file:
            
            img_path= os.path.join(app.config["UPLOAD_F"],file.filename)
            file.save(img_path)
            
            
            results=model(img_path)
            result=results[0]
            
            
            img=cv2.imread(img_path)
            
            for box in result.boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cls=int(box.cls[0])
                
                class_name= model.names[cls] if hasattr(model,"names") else f"Class {cls}"
              
                
                #Draw a bounding box 
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                
                label=f"{class_name}"
                cv2.putText(img,label,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                
                
            processed_p=os.path.join(app.config["PROCCESSED_F"],file.filename)
            success=cv2.imwrite(processed_p,img)
            
            if not success:
                print(f"Error processed image is not save  {processed_p}")
                return render_template("index.html",uploaded_img=None, processed_img=None)
            
            
            upload_img_url=url_for("static",filename=f"uploads/{file.filename}")
            processed_img_url=url_for("serve_prediction",filename=file.filename)
            
            return render_template("index.html",upload_img=upload_img_url,processed_img=processed_img_url)
    
    return render_template("index.html",upload_img=None,processed_img=None)


# Serve processed images correctly
@app.route("/static/Prediction/<filename>")
def serve_prediction(filename):
    return send_from_directory(app.config["PROCCESSED_F"], filename)


if __name__ == "__main__":
    app.run(debug=True)