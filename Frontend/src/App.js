import React from "react";
import ImageUpload from "./ImageUpload";

const App = () => {
    return (<div>
        <div className="bg-image"></div> {/* Background image */}
        <div className="container">
            <h1>Pest Prediction and Pesticide Detection</h1>
            <ImageUpload />
        </div>
        </div>
    );
};

export default App;
