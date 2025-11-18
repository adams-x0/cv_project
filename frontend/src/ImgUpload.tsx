import { useRef, useState } from "react";
import { Upload } from "lucide-react";
import { motion } from "framer-motion";
import React from "react";

interface ImageUploaderProps {
  onImageUpload: (imageUrl: string) => void;
}

export function ImageUploader({ onImageUpload }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [model, setModel] = useState<string>("ViT");
  const [prompt, setPrompt] = useState<string>("");
  const [overlayImage, setOverlayImage] = useState<string | null>(null);
  const [stickers, setStickers] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const BACKEND = (import.meta as any).env.VITE_BACKEND_URL || "http://localhost:8000";

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
    if (!selectedFile) return alert("Please upload an image first!");
    if (model === "ViT" && !prompt)
      return alert('Enter a text prompt for OWL-ViT (e.g., "cat", "car")');

    const formData = new FormData();
    formData.append("image", selectedFile);
    if (model === "ViT") formData.append("prompt", prompt);

    // ✅ Correct backend endpoints
    const endpoint =
      model === "ViT"
        ? `${BACKEND}/segment/owlvit`
        : `${BACKEND}/segment/yolo`;

    try {
      setLoading(true);
      setOverlayImage(null);
      setStickers([]);

      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.error) {
        alert(data.error);
      } else {
        if (data.overlay) {
          setOverlayImage(`http://localhost:8000${data.overlay}`);
        } else {
          setOverlayImage(null);
        }
        
        if (data.stickers && Array.isArray(data.stickers)) {
          setStickers(data.stickers.map((p: string) => `http://localhost:8000${p}`));
        } else {
          setStickers([]);
        }        
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend.");
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
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-sage-100 border border-sage-200 mb-6">
            <span className="text-sm text-sage-700">~Daily Segmentation~</span>
          </div>
          <h1 className="text-6xl mb-4 font-medium text-sage-900">
            Simple Stickers
          </h1>
          <p className="text-lg text-sage-600 max-w-2xl mx-auto mb-10">
            Upload any image, detect objects or search with text, then save as
            stickers!
          </p>

          {/* Upload Area */}
          <div
            className={`relative rounded-3xl p-12 text-center border-2 transition-all cursor-pointer shadow-sm bg-white ${
              isDragging
                ? "border-sage-400 bg-sage-50"
                : "border-sage-200 hover:border-sage-400"
            }`}
            onClick={
              !previewImage ? () => fileInputRef.current?.click() : undefined
            }
            onDragOver={(e) => {
              e.preventDefault();
              if (!previewImage) setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              const file = e.dataTransfer.files[0];
              if (file && file.type.startsWith("image/")) {
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
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 flex items-center justify-center rounded-2xl bg-sage-400 text-white mb-6">
                  <Upload className="size-10" />
                </div>
                <p className="text-xl text-sage-900 mb-2">
                  Drop your image here
                </p>
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

                {/* Model Select + Prompt Input + Button */}
                <div className="flex flex-col sm:flex-row gap-4 items-center">
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="border border-sage-300 rounded-xl px-4 py-2 bg-sage-50 text-sage-800"
                  >
                    <option value="YOLO">YOLOv8-Seg</option>
                    <option value="ViT">OWL-ViT + SAM</option>
                  </select>

                  {model === "ViT" && (
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
                    {loading ? "Processing..." : "Segment"}
                  </button>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* ✅ Display Overlay Result */}
        {overlayImage && (
          <div className="text-center mt-10">
            <h2 className="text-xl text-sage-700 mb-4">Overlay (Boxes + Confidence)</h2>
            <img
              src={overlayImage}
              alt="Overlay Result"
              className="max-h-[60vh] mx-auto rounded-2xl shadow-lg"
            />
          </div>
        )}

        {/* ✅ Display Sticker Result */}
        {stickers.length > 0 && (
          <div className="mt-10">
            <h2 className="text-xl text-sage-700 mb-4 text-center">
              Individual Stickers (Transparent, Cropped)
            </h2>
            <div className="flex flex-wrap justify-center gap-6">
              {stickers.map((stickerUrl, idx) => (
                <div key={idx} className="flex flex-col items-center">
                  <img
                    src={stickerUrl}
                    alt={`Sticker ${idx + 1}`}
                    className="max-h-[40vh] max-w-[200px] object-contain border border-sage-200 rounded-xl bg-sage-50"
                  />
                  <a
                    href={stickerUrl}
                    download={`sticker_${idx + 1}.png`}
                    className="mt-2 inline-block bg-sage-500 hover:bg-sage-600 text-white text-sm px-3 py-1 rounded-xl"
                  >
                    Download
                  </a>
                </div>
              ))}
            </div>
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
