import React, { useState } from "react";
import axios from "axios";

const ImageUpload = () => {
    const [file, setFile] = useState(null);
    const [response, setResponse] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setResponse(null); 
        setError(null); 
    };

    const handleUploadAndPredict = async (e) => {
        e.preventDefault();

        if (!file) {
            setError("Please select a file to upload.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const uploadRes = await axios.post("http://127.0.0.1:5000/upload", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            if (uploadRes.data.path) {
                const predictRes = await axios.post(
                    "http://127.0.0.1:5000/predict-image",
                    { file_path: uploadRes.data.path }, // Pass the file path
                    {
                        headers: {
                            "Content-Type": "application/json",
                        },
                        responseType: "text",
                    }
                );

                setResponse(predictRes.data); 
                setError(null); 
            } else {
                setError("File upload succeeded but no path returned from the server.");
            }
        } catch (err) {
            setError(err.response?.data?.error || "An error occurred during upload or prediction.");
        }
    };

    return (
        <div>
        <div className="container">
            <h2>Upload Image for Prediction</h2>
            <form onSubmit={handleUploadAndPredict}>
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <button type="submit" style={{ marginLeft: "10px" }}>Submit</button>
            </form>
            {response && (
            <div style={{ marginTop: "20px" }}>
                <h3>Prediction Response:</h3>
                    <div style={{
                        backgroundColor: "#f9f9f9",
                        padding: "10px",
                        borderRadius: "5px",
                        border: "1px solid #ccc",
                        maxWidth: "600px"
                    }}>
            {response.split("\n").map((line, index) => (
                <p key={index} style={{ margin: "5px 0", lineHeight: "1.6" }}>
                    {line}
                </p>
            ))}
        </div>
    </div>
)}

            
            {error && (
                <div style={{ marginTop: "20px", color: "red" }}>
                    <h3>Error:</h3>
                    <p>{error}</p>
                </div>
            )}
        </div>
        </div>
    );
};

export default ImageUpload;
