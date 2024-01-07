import { useState } from "react";
import "./Home.css";

const Home = () => {
  const [file, setFile] = useState(null);
  const [isDragging, setIsdragging] = useState(false);
  const [isFileUploaded, setisFileUploaded] = useState(false);
  const [isFileUpload, setisFileUpload] = useState(false);
  const [uploadProgress, setUploadedProgress] = useState(0);
  const [resMsg, setResmsg] = useState("");

  const handleDrop = (e) => {
    e.preventDefault();
    setIsdragging(false);
    const droppedFile = e.dataTransfer.files[0];
    console.log(droppedFile);
    setFile(droppedFile);
    setisFileUpload(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsdragging(true);
  };
  const handleDragLeave = () => {
    setIsdragging(false);
  };

  const handleChange = (e) => {
    console.log(e.target.files);
    setFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (file) {
      const formData = new FormData();
      formData.append("audioFile", file);

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "http://localhost:5000/", true);
      xhr.upload.onprogress = (e) => {
        const progress = (e.target / e.total) * 100;
        setUploadedProgress(progress);
      };

      xhr.onload = () => {
        console.log(JSON.parse(xhr.responseText));
        setResmsg(JSON.parse(xhr.responseText));
        setisFileUploaded(true);
      };
      xhr.send(formData);
    }
  };

  return (
    <div className="Home">
      <div
        className={`UploadFile ${isDragging ? "dragging" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {!isFileUpload ? (
          <>
            <p>Drag and drop a file here, or click to select a file.</p>
            <input type="file" onChange={handleChange} />
          </>
        ) : (
          <>
            <p>Filename: {file.name}</p>
            {!isFileUploaded && (
              <button
                onClick={() => {
                  setisFileUpload(false);
                  setFile(null);
                }}
              >
                cancel
              </button>
            )}
            {/* <input type="button" onClick={setisFileUpload(!isFileUpload)}>
              Cancel
            </input> */}
            {uploadProgress > 0 && (
              <div
                className="progress-bar"
                style={{ width: `${uploadProgress}%` }}
              />
            )}
            {isFileUploaded &&
              (resMsg.status === "Success" ? (
                <>
                  <p className="sucessMsg">{resMsg.message}</p>
                  <button
                    onClick={() => {
                      setisFileUpload(false);
                      setFile(null);
                    }}
                  >
                    Continue
                  </button>
                </>
              ) : (
                <p className="errorMsg">{resMsg.message}</p>
              ))}
          </>
        )}
      </div>
      <button onClick={handleUpload} disabled={!file}>
        Upload
      </button>
    </div>
  );
};

export default Home;
