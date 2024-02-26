import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { AgGridReact } from "ag-grid-react"; // AG Grid Component
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import ClipLoader from "react-spinners/ClipLoader";
import "./Test.css";

const Test = () => {
  const [data, setData] = useState(null);
  const [loading, setIsLoading] = useState(true);
  const [colDefs, setColDefs] = useState([
    { headerName: "Sno", valueGetter: "node.rowIndex + 1", width: 100 },
    { field: "Audio", flex: 1 },
    { field: "SpeakerLabel", flex: 1 },
    { field: "StartTime", flex: 1 },
    { field: "EndTime", flex: 1 },
  ]);

  const location = useLocation();
  console.log(location.state?.filename);
  useEffect(() => {
    setIsLoading(true);
    fetch("http://localhost:5000/getResult", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ filename: location.state?.filename }),
    })
      .then((res) => res.json())
      .then((res) => {
        console.log(res);
        setData(res.data);
        setIsLoading(false);
      });
  }, []);

  return (
    <div
      className="ag-theme-quartz" // applying the grid theme
      style={{ height: "100vh", width: "100%" }} // the grid will fill the size of the parent container
    >
      {loading ? (
        <div className="loader">
          <ClipLoader
            loading={loading}
            size={150}
            aria-label="Loading Spinner"
            data-testid="loader"
          />
          <h1>Training the Model and Performing Diarisation</h1>
        </div>
      ) : (
        <AgGridReact rowData={data} columnDefs={colDefs} />
      )}
    </div>
  );
};

export default Test;
