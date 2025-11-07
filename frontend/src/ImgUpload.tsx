import { useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';
import React from 'react';

interface ImageUploaderProps {
  onImageUpload: (imageUrl: string) => void;
}

export function ImageUploader({ onImageUpload }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [model, setModel] = useState<string>('ViT');
  const [prompt, setPrompt] = useState<string>('');
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [stickerImage, setStickerImage] = useState<string | null>(null); // ✅ Kept EXACTLY as you requested
  const [loading, setLoading] = useState(false);

  // Handle file select
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageUrl = event.target?.result as string;
        setPreviewImage(imageUrl);
        onImageUpload(imageUrl);
      };
      reader.readAsDataURL(file);
    }
  };

  // Send image + prompt to FastAPI backend
  const processImage = async () => {
    if (!selectedFile) return alert('Please upload an image first!');
    if (model === 'ViT' && !prompt) {
      return alert('Enter a text prompt for OWL-ViT (e.g., "cat", "car")');
    }

    const formData = new FormData();
    formData.append('image', selectedFile);
    if (model === 'ViT') formData.append('prompt', prompt);

    const endpoint =
      model === 'ViT'
        ? 'http://localhost:8000/segment/owlvit'
        : 'http://localhost:8000/api/segment';

    try {
      setLoading(true);
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.overlay) {
        setResultImage(`http://localhost:8000${data.overlay}`);
      }
      if (data.sticker) {
        setStickerImage(`http://localhost:8000${data.sticker}`);
      }
      if (data.error) {
        alert(data.error);
      }
    } catch (err) {
      console.error(err);
      alert('Error connecting to backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-sage-50 text-sage-900">
      <div className="max-w-5xl mx-auto px-6 pt-20 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >
          {/* Title + Subtitle */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-sage-100 border border-sage-200 mb-6">
            <span className="text-sm text-sage-700">~Daily Segmentation~</span>
          </div>
          <h1 className="text-6xl mb-4 font-medium text-sage-900">Simple Stickers</h1>
          <p className="text-lg text-sage-600 max-w-2xl mx-auto mb-10">
            Upload any image, detect objects or search with text, then save as stickers!
          </p>

          {/* Upload Area */}
          <div
            className={`relative rounded-3xl p-12 text-center border-2 transition-all cursor-pointer shadow-sm bg-white ${
              isDragging ? 'border-sage-400 bg-sage-50' : 'border-sage-200 hover:border-sage-400'
            }`}
            onClick={!previewImage ? () => fileInputRef.current?.click() : undefined}
            onDragOver={(e) => {
              e.preventDefault();
              if (!previewImage) setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              const file = e.dataTransfer.files[0];
              if (file && file.type.startsWith('image/')) {
                setSelectedFile(file);
                const reader = new FileReader();
                reader.onload = (event) => {
                  const url = event.target?.result as string;
                  setPreviewImage(url);
                  onImageUpload(url);
                };
                reader.readAsDataURL(file);
              }
            }}
          >
            {!previewImage ? (
              // Before image is uploaded
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 flex items-center justify-center rounded-2xl bg-sage-400 text-white mb-6">
                  <Upload className="size-10" />
                </div>
                <p className="text-xl text-sage-900 mb-2">Drop your image here</p>
                <p className="text-sm text-sage-600">or click to browse • JPG, PNG, WEBP • Max 10MB</p>
              </div>
            ) : (
              // After image is uploaded
              <div className="flex flex-col items-center gap-8">
                <img
                  src={previewImage}
                  alt="Uploaded"
                  className="max-h-[50vh] w-auto rounded-2xl object-contain shadow-md border border-sage-200"
                />

                {/* Model Select + Prompt Input + Button */}
                <div className="flex flex-col sm:flex-row gap-4 items-center">
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="border border-sage-300 rounded-xl px-4 py-2 bg-sage-50 text-sage-800"
                  >
                    <option value="YOLO">YOLO</option>
                    <option value="ViT">OWL-ViT + SAM</option>
                  </select>

                  {model === 'ViT' && (
                    <input
                      type="text"
                      placeholder="Search object (e.g., car, cat)"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="border border-sage-300 rounded-xl px-4 py-2 bg-sage-50 text-sage-800 w-64"
                    />
                  )}

                  <button
                    onClick={processImage}
                    disabled={loading}
                    className="px-6 py-2 bg-sage-500 text-white rounded-xl hover:bg-sage-600 disabled:opacity-50"
                  >
                    {loading ? 'Processing...' : 'Segment'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* ✅ Display Overlay Result */}
        {resultImage && (
          <div className="text-center mt-10">
            <h2 className="text-xl text-sage-700 mb-4">Overlay Result</h2>
            <img src={resultImage} alt="Overlay Result" className="max-h-[60vh] mx-auto rounded-2xl shadow-lg" />
          </div>
        )}

        {/* ✅ Display Sticker Result */}
        {stickerImage && (
          <div className="text-center mt-10">
            <h2 className="text-xl text-sage-700 mb-4">Sticker (Transparent)</h2>
            <img src={stickerImage} alt="Sticker Result" className="max-h-[50vh] mx-auto object-contain" />
            <a
              href={stickerImage}
              download="sticker.png"
              className="mt-4 inline-block bg-sage-500 hover:bg-sage-600 text-white px-4 py-2 rounded-xl"
            >
              Download Sticker
            </a>
          </div>
        )}
      </div>

      {/* Hidden input for file selection */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  );
}
