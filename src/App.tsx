import React from 'react';
import { ImageUploader } from './ImgUpload';

function App() {
  return (
    <div className="min-h-screen bg-sage-50 text-sage-900 flex items-center justify-center">
      <ImageUploader onImageUpload={() => {}} />
    </div>
  );
}

export default App;

