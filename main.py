from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException
import shutil
import os
import data_preprocessing


app = FastAPI()

@app.get("/status/")
async def status():
    return {"message": "Welcome to the Audio Classification API"}


@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    if not file:
        return HTTPException(status_code=400, detail="No file provided")
    elif not file.filename.endswith(".m4a"):
        return HTTPException(status_code=400, detail="Only .wav files are allowed")
    file_path = os.path.join("audio_files", file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            print(f"File {file.filename} saved successfully")
        return {"message": "File uploaded successfully", "filename": file.filename, "file_path": file_path}  # Successful response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    
@app.get("/predict/") 
async def predict(file_name: str):
    data_dir = "audio_files"
    output_label = data_preprocessing.get_output_label(data_dir, file_name)
    print(output_label, " output_label")
    return {"message": output_label}