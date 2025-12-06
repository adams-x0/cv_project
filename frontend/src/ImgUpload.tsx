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

  // model can be: "YOLO", "ViT" (OWL-ViT), "RTDETR", "GROUND" (GroundingDINO)
  const [model, setModel] = useState<string>("ViT");

  const [prompt, setPrompt] = useState<string>("");
  const [overlayImage, setOverlayImage] = useState<string | null>(null);
  const [stickers, setStickers] = useState<string[]>([]);
  const [labels, setLabels] = useState<string[]>([]);     // ⭐ NEW
  const [loading, setLoading] = useState(false);

  const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

  // ----------------------------------------------
  // Handle File Selection (reset outputs)
  // ----------------------------------------------
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setOverlayImage(null);
      setStickers([]);
      setLabels([]); // ⭐ Clear labels
      setPrompt("");

      const reader = new FileReader();
      reader.onload = (event) => {
        const imageUrl = event.target?.result as string;
        setPreviewImage(imageUrl);
        onImageUpload(imageUrl);
      };
      reader.readAsDataURL(file);
    }
  };

  // ----------------------------------------------
  // Backend Call
  // ----------------------------------------------
  const processImage = async () => {
    if (!selectedFile) {
      alert("Please upload an image first!");
      return;
    }

    // Only OWL-ViT and GroundingDINO require a prompt
    const needsPrompt = model === "ViT" || model === "GROUND";
    if (needsPrompt && !prompt.trim()) {
      alert("Enter a text prompt (e.g., 'a red car').");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);
    if (needsPrompt) {
      formData.append("prompt", prompt);
    }

    const endpoint =
      model === "ViT"
        ? `${BACKEND}/segment/owlvit`
        : model === "RTDETR"
        ? `${BACKEND}/segment/rtdetr`
        : model === "GROUND"
        ? `${BACKEND}/segment/grounded`
        : model === "MASK2FORMER"
        ? `${BACKEND}/segment/mask2former`
        : `${BACKEND}/segment/yolo`;

    try {
      setLoading(true);
      setOverlayImage(null);
      setStickers([]);
      setLabels([]);   // ⭐ Reset labels

      const res = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();

      // Handle "object not found" for *text-driven* models
      if (needsPrompt) {
        const noOverlay = !data.overlay;
        const noStickers = !data.stickers || data.stickers.length === 0;

        if (noOverlay && noStickers) {
          alert("Object not found in the image.");
          return;
        }
      }

      // normal behavior
      if (data.overlay) setOverlayImage(`${BACKEND}${data.overlay}`);
      if (data.stickers) {
        setStickers(data.stickers.map((p: string) => `${BACKEND}${p}`));
      }
      if (data.labels) {
        setLabels(data.labels); // ⭐ Store labels
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-sage-50 to-sage-300 text-sage-900">
      <div className="max-w-5xl mx-auto px-6 pt-20 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-sage-100 border border-sage-200 mb-6 shadow-sm">
            <span className="text-sm text-sage-700">~Daily Segmentation~</span>
          </div>

          <h1 className="text-6xl mb-4 font-semibold tracking-tight text-sage-900">
            Simple Stickers
          </h1>

          <p className="text-lg text-sage-600 max-w-2xl mx-auto mb-10">
            Upload any image, detect objects, or search with text — then save
            as high-quality stickers.
          </p>

          {/* Upload Container */}
          <div
            className={`relative rounded-3xl p-12 text-center 
            backdrop-blur-2xl bg-white/50 shadow-2xl border border-white/40 
            transition-all duration-300 cursor-pointer 
            ${isDragging ? "scale-[1.02]" : "hover:shadow-xl"}
          `}
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
              if (file) {
                setSelectedFile(file);
                setOverlayImage(null);
                setStickers([]);
                setLabels([]);

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
            {/* No image yet */}
            {!previewImage ? (
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 flex items-center justify-center rounded-2xl bg-sage-500 text-white shadow-lg mb-6">
                  <Upload className="size-10" />
                </div>

                <p className="text-xl font-medium text-sage-900 mb-2">
                  Drop your image here
                </p>
                <p className="text-sm text-sage-600">
                  or click to browse • JPG, PNG, WEBP • Max 10MB
                </p>
              </div>
            ) : (
              <>
                {/* Image preview */}
                <motion.img
                  src={previewImage}
                  alt="Uploaded"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4 }}
                  className="max-h-[50vh] mx-auto rounded-2xl shadow-xl border border-sage-200 object-contain"
                />

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-4 px-4 py-1 rounded-xl bg-white/70 border border-sage-300 text-sm shadow-sm hover:bg-white transition"
                >
                  Change Image
                </button>

                {/* Controls */}
                <div className="flex flex-col sm:flex-row gap-4 mt-8 items-center justify-center">
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="border border-sage-300 rounded-xl px-4 py-2 bg-white shadow-sm focus:ring-2 focus:ring-sage-400"
                  >
                    <option value="YOLO">YOLOv12-Seg</option>
                    <option value="MASK2FORMER">Mask2Former</option>
                    <option value="RTDETR">RT-DETR + SAM2</option>
                    <option value="ViT">OWL-ViT + SAM2</option>
                    <option value="GROUND">GroundingDINO + SAM2</option>
                  </select>

                  {(model === "ViT" || model === "GROUND") && (
                    <input
                      type="text"
                      placeholder="Search object (e.g., red car, dog)"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="border border-sage-300 rounded-xl px-4 py-2 bg-white shadow-sm w-64 focus:ring-2 focus:ring-sage-400"
                    />
                  )}

                  <button
                    onClick={processImage}
                    disabled={loading}
                    className="px-6 py-2 rounded-xl text-white bg-sage-600 hover:bg-sage-700 shadow-md hover:shadow-lg transition disabled:opacity-50"
                  >
                    {loading ? "Processing..." : "Segment"}
                  </button>
                </div>
              </>
            )}
          </div>
        </motion.div>

        {/* Overlay Image */}
        {overlayImage && (
          <div className="text-center mt-10">
            <h2 className="text-xl text-sage-700 mb-4 font-semibold">
              Overlay (Boxes + Confidence)
            </h2>
            <img
              src={overlayImage}
              className="max-h-[60vh] mx-auto rounded-2xl shadow-xl"
            />
          </div>
        )}

        {/* Sticker Results */}
        {stickers.length > 0 && (
          <div className="mt-10">
            <h2 className="text-xl text-sage-700 mb-6 text-center font-semibold">
              Individual Stickers
            </h2>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-8 justify-center">
              {stickers.map((s, idx) => (
                <motion.div
                  key={idx}
                  whileHover={{ scale: 1.05 }}
                  className="flex flex-col items-center"
                >
                  <img
                    src={s}
                    className="max-h-[40vh] max-w-[200px] rounded-xl shadow-md bg-white border border-sage-200 object-contain"
                    alt={`Sticker ${idx + 1}`}
                  />

                  {/* ⭐ LABEL UNDER STICKER */}
                  <p className="mt-2 text-sm text-sage-700 font-medium">
                    {labels[idx] ?? labels[0] ?? prompt ?? "Object"}
                  </p>

                  <a
                    href={s}
                    download={`sticker_${idx + 1}.png`}
                    className="mt-2 px-3 py-1 rounded-xl bg-sage-600 text-white text-sm hover:bg-sage-700 shadow-sm"
                  >
                    Download
                  </a>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Hidden Input */}
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
