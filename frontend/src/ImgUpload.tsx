import React, { useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';
import {useNavigate} from 'react-router-dom';


interface ImageUploaderProps {
  onImageUpload: (imageUrl: string) => void;
};

// Base backend address
// We will build URLs like http://localhost:8000/api/segment
const BACKEND = "http://localhost:8000";


export function ImageUploader({ onImageUpload }: ImageUploaderProps) {
  // Reference to the hidden file input element
  const fileInputRef = useRef<HTMLInputElement>(null);

  // UI states
  const [isDragging, setIsDragging] = useState(false);    // Drag and drop// Drag & drop hover effect
  const [previewImage, setPreviewImage] = useState<string | null>(null);    // Shown preview

  // Actual image file object (needed for upload)
  const [fileObj, setFileObj] = useState<File | null>(null);

  // Whether segmentation request is currently running
  const [loading, setLoading] = useState(false);

  // Used to move to the results page after finishing segmentation
  const navigate = useNavigate();


  // ---------------------------------------------------------
  // Handle clicking "choose file" or dropping a file
  // Save image for upload + generate preview
  // ---------------------------------------------------------
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileObj(file); // keep original file for upload

      const reader = new FileReader();
      reader.onload = (event) => {
        const imageUrl = event.target?.result as string;
        setPreviewImage(imageUrl);
        onImageUpload(imageUrl);      // Parent callback (optional preview use)
      };
      reader.readAsDataURL(file);     // Convert file → Base64 for display
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setFileObj(file); // keep original file for upload
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageUrl = event.target?.result as string;
        setPreviewImage(imageUrl);
        onImageUpload(imageUrl);
      };
      reader.readAsDataURL(file);   // Convert file → Base64 for display
    }
  };


  // ---------------------------------------------------------
  // Send the uploaded image to backend → run segmentation
  // Save returned overlay & stickers in localStorage
  // Navigate to results page to display them
  // ---------------------------------------------------------
  async function handleRun() {
    if (!fileObj) {
      alert('Please upload an image first.');
      return;
    }

    try {
      setLoading(true);

      // FormData allows file upload in HTTP request
      const form = new FormData();
      form.append('image', fileObj);    // Backend expects the field name `image`

      const res = await fetch(`${BACKEND}/api/segment`, {
        method: 'POST',
        body: form, // ✅ do not set Content-Type manually
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(text || `Segmentation failed (${res.status})`);
      }

      const data: { overlay_url: string | null; sticker_urls?: string[] } = await res.json();

      // Convert returned relative URLs → Full URLs usable by browser
      const overlayFull = data.overlay_url ? `${BACKEND}${data.overlay_url}` : "";
      const stickersFull = (data.sticker_urls ?? []).map(p => `${BACKEND}${p}`);

      // Store so the results page can load them
      localStorage.setItem('overlayUrl', overlayFull);
      localStorage.setItem('stickerUrls', JSON.stringify(stickersFull));

      // Redirect to results page
      navigate('/run_yolo');
    } catch (err: any) {
      console.error('Segmentation error:', err);
      alert(err?.message || 'Segmentation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  // ---------------------------------------------------------
  // UI Layout & JSX Rendering
  // ---------------------------------------------------------
  return (
    <div className="min-h-screen flex flex-col bg-sage-50 text-sage-900">
      <div className="max-w-5xl mx-auto px-6 pt-20 pb-12">

        {/* Animated fade-in container */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-16"
        >

          {/* Header Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-sage-100 border border-sage-200 mb-6">
            <span className="text-sm text-sage-700">~Daily Segmentation~</span>
          </div>

          <h1 className="text-6xl mb-4 font-medium text-sage-900">Simple Stickers</h1>

          <p className="text-lg text-sage-600 max-w-2xl mx-auto mb-10">
            Upload any image and instantly detect objects/ search for items in image and it will be selected with the option to save as a sticker!
          </p>

          {/* Upload Box */}
          <div
            className={`relative rounded-3xl p-12 text-center border-2 transition-all cursor-pointer shadow-sm bg-white ${
              isDragging
                ? 'border-sage-400 bg-sage-50 scale-[1.01]'
                : 'border-sage-200 hover:border-sage-400'
            }`}
            onDrop={!previewImage ? handleDrop : undefined}
            onDragOver={!previewImage ? (e) => { e.preventDefault(); setIsDragging(true); } : undefined}
            onDragLeave={!previewImage ? () => setIsDragging(false) : undefined}
            onClick={!previewImage ? () => fileInputRef.current?.click() : undefined}
          >
            {/* If no preview yet → show upload instructions */}
            {!previewImage ? (
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 flex items-center justify-center rounded-2xl bg-sage-400 text-white mb-6">
                  <Upload className="size-10" />
                </div>
                <p className="text-xl text-sage-900 mb-2">Drop your image here</p>
                <p className="text-sm text-sage-600">
                  or click to browse • JPG, PNG, WEBP • Max 10MB
                </p>
              </div>
            ) : (
               // Show preview & model options once image selected 
              <div className="flex flex-col items-center gap-8">
                <img
                  src={previewImage}
                  alt="Uploaded"
                  className="max-h-[50vh] w-auto rounded-2xl object-contain shadow-md border border-sage-200"
                />
                  
                {/* Placeholder UI for model selection & search */}
                <div className="flex flex-col sm:flex-row gap-4 items-center">
                  <select aria-label="Select segmentation model" title="Select segmentation model" className="border border-sage-300 rounded-xl px-4 py-2 bg-sage-50 text-sage-800 focus:border-sage-500 focus:ring-1 focus:ring-sage-500">
                    <option>Select segmentation model</option>
                    <option>YOLO</option>
                    <option>ViT</option>
                  </select>
                  <input
                    type="text"
                    placeholder="Search object (e.g., car, cat)"
                    className="border border-sage-300 rounded-xl px-4 py-2 bg-sage-50 text-sage-800 w-64 focus:border-sage-500 focus:ring-1 focus:ring-sage-500"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Run Segmentation Button (only appears after image uploaded) */}
          {previewImage && (
            <div className="mt-8 flex justify-center">
              <button
                onClick={handleRun}
                disabled={loading}
                className="inline-flex items-center px-6 py-3 rounded-xl bg-sage-600 text-white text-lg font-medium transition-all shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Run Segmentation'}      
              </button>
            </div>
          )}

        </motion.div>
      </div>

       {/* Hidden file input (triggered by clicking upload box) */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
        aria-label="Upload image"
        title="Choose an image file to upload"
      />
    </div>
  );
}
