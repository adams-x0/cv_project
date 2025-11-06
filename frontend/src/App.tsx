import React from 'react';
import { ImageUploader } from '@/ImgUpload';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import RunYoloSegmentationPage  from '@/pages/RunYoloSegmentationPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-sage-50 text-sage-900 flex items-center justify-center">
        <Routes>
          <Route path="/" element={<ImageUploader onImageUpload={() => { }} />} />
          <Route path="/run_yolo" element={<RunYoloSegmentationPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;

