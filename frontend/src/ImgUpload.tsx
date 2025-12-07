// ImgUpload.tsx
import { useRef, useState } from "react";
import { Upload } from "lucide-react";
import { motion } from "framer-motion";
import React from "react";
import { StickerCanvasEditor } from "./StickerCanvasEditor";

interface ImageUploaderProps {
  onImageUpload: (imageUrl: string) => void;
}

export function ImageUploader({ onImageUpload }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [isDragging, setIsDragging] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [model, setModel] = useState<string>("YOLO");
  const [prompt, setPrompt] = useState<string>("");

  const [overlayImage, setOverlayImage] = useState<string | null>(null);
  const [stickers, setStickers] = useState<string[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

  // Background placement editor
  const [bgImage, setBgImage] = useState<string | null>(null);
  const [activeSticker, setActiveSticker] = useState<string | null>(null);
  const [canvasMode, setCanvasMode] = useState(false);

  // Results from /segment/all
  const [allResults, setAllResults] = useState<any | null>(null);

  // ----------------------------------------
  // File Selection
  // ----------------------------------------
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setOverlayImage(null);
    setStickers([]);
    setLabels([]);
    setPrompt("");
    setAllResults(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      const url = event.target?.result as string;
      setPreviewImage(url);
      onImageUpload(url);
    };
    reader.readAsDataURL(file);
  };

  // ----------------------------------------
  // Process Single Model
  // ----------------------------------------
  const processImage = async () => {
    if (!selectedFile) return alert("Upload an image first!");

    const needsPrompt = model === "ViT" || model === "GROUND";
    if (needsPrompt && !prompt.trim()) return alert("Enter text prompt");

    const formData = new FormData();
    formData.append("image", selectedFile);
    if (needsPrompt) formData.append("prompt", prompt);

    const endpoint =
      model === "ViT"
        ? `${BACKEND}/segment/owlvit`
        : model === "GROUND"
        ? `${BACKEND}/segment/grounded`
        : model === "RTDETR"
        ? `${BACKEND}/segment/rtdetr`
        : model === "MASK2FORMER"
        ? `${BACKEND}/segment/mask2former`
        : model === "TRAINEDYOLO"
        ? `${BACKEND}/segment/trained_yolo`
        : `${BACKEND}/segment/yolo`;

    try {
      setLoading(true);
      setAllResults(null);
      setOverlayImage(null);
      setStickers([]);
      setLabels([]);

      const res = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();

      if (data.overlay) setOverlayImage(`${BACKEND}${data.overlay}`);
      if (data.stickers)
        setStickers(data.stickers.map((s: string) => `${BACKEND}${s}`));
      if (data.labels) setLabels(data.labels);
    } catch (e) {
      console.error(e);
      alert("Backend error");
    } finally {
      setLoading(false);
    }
  };

  // ----------------------------------------
  // Process ALL Models
  // ----------------------------------------
  const processAll = async () => {
    if (!selectedFile) return alert("Upload an image first!");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      setLoading(true);
      setAllResults(null);

      const res = await fetch(`${BACKEND}/segment/all`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      // Add open flag for accordion
      const opened = Object.fromEntries(
        Object.entries(data).map(([k, v]) => {
          if (typeof v === "object") return [k, { ...v, open: false }];
          return [k, v];
        })
      );

      setAllResults(opened);
    } catch (err) {
      console.error(err);
      alert("Backend error");
    } finally {
      setLoading(false);
    }
  };

  // ----------------------------------------
  // UI
  // ----------------------------------------
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-sage-50 to-sage-300 text-sage-900">
      <div className="max-w-5xl mx-auto px-6 pt-20 pb-12">

        {/* HEADER */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-sage-100 border border-sage-200 mb-6 shadow-sm">
            <span className="text-sm text-sage-700">~Daily Segmentation~</span>
          </div>

          <h1 className="text-6xl mb-4 font-semibold">Simple Stickers</h1>
          <p className="text-lg text-sage-600 max-w-2xl mx-auto mb-10">
            Upload an image, segment objects, and create stickers.
          </p>

          {/* UPLOAD BOX */}
          <div
            className={`relative rounded-3xl p-12 text-center backdrop-blur-xl bg-white/50 shadow-2xl border border-white/40 transition-all ${
              isDragging ? "scale-[1.02]" : "hover:shadow-xl"
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
              if (!file) return;

              setSelectedFile(file);
              setAllResults(null);
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
            }}
          >
            {!previewImage ? (
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 flex items-center justify-center rounded-2xl bg-sage-500 text-white shadow-lg mb-6">
                  <Upload className="size-10" />
                </div>
                <p className="text-xl font-medium">Drop your image here</p>
                <p className="text-sm text-sage-600">or click to browse • JPG, PNG, WEBP</p>
              </div>
            ) : (
              <>
                <motion.img
                  src={previewImage}
                  className="max-h-[50vh] mx-auto rounded-2xl shadow-xl border border-sage-200 object-contain"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                />

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-4 px-4 py-1 rounded-xl bg-white/70 border border-sage-300 text-sm shadow-sm hover:bg-white"
                >
                  Change Image
                </button>

                <div className="flex flex-col sm:flex-row gap-4 mt-8 items-center justify-center">
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="border border-sage-300 rounded-xl px-4 py-2 bg-white shadow-sm"
                  >
                    <option value="YOLO">YOLOv12-Seg</option>
                    <option value="TRAINEDYOLO">Trained YOLO</option>
                    <option value="MASK2FORMER">Mask2Former</option>
                    <option value="RTDETR">RT-DETR + SAM2</option>
                    <option value="ViT">OWL-ViT + SAM2</option>
                    <option value="GROUND">GroundingDINO + SAM2</option>
                  </select>

                  {(model === "ViT" || model === "GROUND") && (
                    <input
                      type="text"
                      placeholder="Search object (e.g., red car)"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="border border-sage-300 rounded-xl px-4 py-2 bg-white w-64 shadow-sm"
                    />
                  )}

                  <button
                    onClick={processImage}
                    disabled={loading}
                    className="px-6 py-2 rounded-xl text-white bg-sage-600 hover:bg-sage-700 shadow-md disabled:opacity-50"
                  >
                    {loading ? "Processing..." : "Segment"}
                  </button>

                  <button
                    onClick={processAll}
                    disabled={loading}
                    className="px-6 py-2 rounded-xl text-white bg-sage-800 hover:bg-sage-900 shadow-md disabled:opacity-50"
                  >
                    {loading ? "Processing..." : "Segment With ALL Models"}
                  </button>
                </div>
              </>
            )}
          </div>
        </motion.div>

        {/* SINGLE MODEL OVERLAY */}
        {overlayImage && (
          <div className="text-center mt-10">
            <h2 className="text-xl font-semibold text-sage-700 mb-4">
              Overlay
            </h2>
            <img
              src={overlayImage}
              className="max-h-[60vh] mx-auto rounded-2xl shadow-xl"
            />
          </div>
        )}

        {/* Single Model Stickers */}
        {stickers.length > 0 && (
          <div className="mt-10">
            <h2 className="text-xl text-sage-700 text-center font-semibold mb-6">
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
                  />

                  <p className="mt-2 text-sm text-sage-700 font-medium">
                    {labels[idx] ?? "Object"}
                  </p>

                  {/* Download */}
                  <a
                    href={`${BACKEND}/download/${s.split("/").pop()}`}
                    className="mt-2 px-3 py-1 rounded-xl bg-sage-600 text-white hover:bg-sage-700 text-sm"
                  >
                    Download
                  </a>

                  {/* Add to Photo */}
                  {bgImage && (
                    <button
                      onClick={() => {
                        setActiveSticker(s);
                        setCanvasMode(true);
                      }}
                      className="mt-2 px-3 py-1 rounded-xl bg-sage-500 text-white hover:bg-sage-600 text-sm"
                    >
                      Add to Photo
                    </button>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* ======================= */}
        {/*   ALL MODELS ACCORDION */}
        {/* ======================= */}
        {allResults && (
          <div className="mt-16">
            <h2 className="text-2xl text-center text-sage-700 font-semibold mb-8">
              All Models — Comparison View
            </h2>

            {Object.entries(allResults).map(([key, result]: any) => {
              if (key === "message") return null;

              return (
                <div
                  key={key}
                  className="mb-6 border border-sage-300 rounded-xl bg-white shadow"
                >
                  {/* ACCORDION HEADER */}
                  <button
                    className="w-full flex justify-between items-center px-5 py-3 bg-sage-100 hover:bg-sage-200 rounded-xl"
                    onClick={() =>
                      setAllResults((prev: any) => ({
                        ...prev,
                        [key]: { ...prev[key], open: !prev[key].open },
                      }))
                    }
                  >
                    <span className="font-bold text-lg text-sage-800">
                      {key.toUpperCase()}
                    </span>
                    <span>{result.open ? "▲" : "▼"}</span>
                  </button>

                  {/* ACCORDION BODY */}
                  {result.open && (
                    <div className="p-5">

                      {/* Overlay */}
                      {result.overlay && (
                        <img
                          src={`${BACKEND}${result.overlay}`}
                          className="max-h-[55vh] mx-auto rounded-xl shadow-md"
                        />
                      )}

                      {/* Stickers */}
                      {result.stickers?.length > 0 && (
                        <>
                          <h3 className="font-semibold text-sage-700 mt-6 mb-3">
                            Stickers
                          </h3>

                          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
                            {result.stickers.map((st: string, i: number) => (
                              <div
                                key={i}
                                className="flex flex-col items-center"
                              >
                                <img
                                  src={`${BACKEND}${st}`}
                                  className="max-h-[30vh] rounded-xl border shadow bg-white object-contain"
                                />

                                <p className="text-sm text-sage-700 mt-1">
                                  {result.labels?.[i] ?? "Object"}
                                </p>

                                {/* DOWNLOAD */}
                                <a
                                  href={`${BACKEND}/download/${st.split("/").pop()}`}
                                  className="mt-2 px-3 py-1 rounded-xl bg-sage-600 text-white text-sm hover:bg-sage-700"
                                >
                                  Download
                                </a>


                                {/* ADD TO PHOTO */}
                                {bgImage && (
                                  <button
                                    onClick={() => {
                                      setActiveSticker(`${BACKEND}${st}`);
                                      setCanvasMode(true);
                                    }}
                                    className="mt-2 px-3 py-1 rounded-xl bg-sage-500 text-white text-sm hover:bg-sage-600"
                                  >
                                    Add to Photo
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* BACKGROUND UPLOADER */}
        <div className="text-center mt-14">
          <h2 className="text-xl text-sage-700 font-semibold mb-2">
            Load Background Image for Sticker Placement
          </h2>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (!file) return;

              const reader = new FileReader();
              reader.onload = (event) =>
                setBgImage(event.target?.result as string);
              reader.readAsDataURL(file);
            }}
            className="border p-2 rounded-xl bg-white shadow-sm"
          />
        </div>
      </div>

      {/* CANVAS EDITOR */}
      {canvasMode && bgImage && activeSticker && (
        <StickerCanvasEditor
          bg={bgImage}
          sticker={activeSticker}
          onClose={() => setCanvasMode(false)}
        />
      )}

      {/* HIDDEN FILE INPUT */}
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
