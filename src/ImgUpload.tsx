import { useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';
import React from 'react';


interface ImageUploaderProps {
  onImageUpload: (imageUrl: string) => void;
};



export function ImageUploader({ onImageUpload }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);


  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageUrl = event.target?.result as string;
        setPreviewImage(imageUrl);
        onImageUpload(imageUrl);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageUrl = event.target?.result as string;
        setPreviewImage(imageUrl);
        onImageUpload(imageUrl);
      };
      reader.readAsDataURL(file);
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
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-sage-100 border border-sage-200 mb-6">
        <span className="text-sm text-sage-700">~Daily Segmentation~</span>
      </div>

      <h1 className="text-6xl mb-4 font-medium text-sage-900">Simple Stickers</h1>

      <p className="text-lg text-sage-600 max-w-2xl mx-auto mb-10">
        Upload any image and instantly detect objects/ search for items in image and it will be selected with the option to save as a sticker!
      </p>

      {/* Upload Area */}
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
          <div className="flex flex-col items-center gap-8">
            <img
              src={previewImage}
              alt="Uploaded"
              className="max-h-[50vh] w-auto rounded-2xl object-contain shadow-md border border-sage-200"
            />
            <div className="flex flex-col sm:flex-row gap-4 items-center">
              <select className="border border-sage-300 rounded-xl px-4 py-2 bg-sage-50 text-sage-800 focus:border-sage-500 focus:ring-1 focus:ring-sage-500">
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
    </motion.div>
  </div>

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
