// StickerCanvasEditor.tsx
import { useEffect, useRef, useState } from "react";

export function StickerCanvasEditor({ bg, sticker, onClose }: any) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Sticker position + scale
  const [pos, setPos] = useState({ x: 100, y: 100 });
  const [scale, setScale] = useState(1);

  // ⭐ Movement speed (user-controlled)
  const [speed, setSpeed] = useState(100); // default

  // ------------------------------------------------
  // Draw Background + Sticker
  // ------------------------------------------------
    useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d")!;
    const bgImg = new Image();
    const stImg = new Image();

    // IMPORTANT: allow export!
    bgImg.crossOrigin = "anonymous";
    stImg.crossOrigin = "anonymous";

    bgImg.src = bg;
    stImg.src = sticker;

    bgImg.onload = () => {
        canvas.width = bgImg.width;
        canvas.height = bgImg.height;

        const draw = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(bgImg, 0, 0);

        const w = stImg.width * scale;
        const h = stImg.height * scale;
        ctx.drawImage(stImg, pos.x, pos.y, w, h);
        };

        stImg.onload = draw;
    };
    }, [bg, sticker, pos, scale]);


  // ------------------------------------------------
  // ⭐ Keyboard Movement ONLY (Arrow Keys)
  // ------------------------------------------------
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      let step = speed;
      if (e.shiftKey) step = speed * 3; // Fast mode

      if (e.key === "ArrowUp") {
        setPos((p) => ({ ...p, y: p.y - step }));
      } else if (e.key === "ArrowDown") {
        setPos((p) => ({ ...p, y: p.y + step }));
      } else if (e.key === "ArrowLeft") {
        setPos((p) => ({ ...p, x: p.x - step }));
      } else if (e.key === "ArrowRight") {
        setPos((p) => ({ ...p, x: p.x + step }));
      }
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [speed]);

  // ------------------------------------------------
  // Save final image
  // ------------------------------------------------
    const download = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob((blob) => {
        if (!blob) return;

        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "composited_image.png";
        a.click();
        URL.revokeObjectURL(url);
    }, "image/png");
    };


  // ------------------------------------------------
  // UI
  // ------------------------------------------------
  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-6">
      <div className="bg-white rounded-2xl shadow-2xl p-6 flex flex-col items-center gap-5 max-w-[90vw]">

        <canvas
          ref={canvasRef}
          className="border rounded-xl shadow-xl max-w-[85vw] max-h-[60vh]"
        />

        {/* ⭐ Movement speed input */}
        <div className="flex items-center gap-3">
          <label className="text-sage-700 font-semibold text-sm">
            Movement Speed (px):
          </label>

          <input
            type="number"
            min={1}
            max={200}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-20 px-2 py-1 border rounded-lg text-center shadow-sm"
          />
        </div>

        <p className="text-xs text-gray-600">
          Use <b>↑ ↓ ← →</b> to move • Hold <b>Shift</b> for fast mode
        </p>

        <div className="flex gap-4 mt-4">
          <button
            onClick={() => setScale(scale + 0.1)}
            className="px-4 py-2 bg-sage-500 text-white rounded-xl"
          >
            Zoom +
          </button>

          <button
            onClick={() => setScale(Math.max(0.1, scale - 0.1))}
            className="px-4 py-2 bg-sage-500 text-white rounded-xl"
          >
            Zoom -
          </button>

          <button
            onClick={download}
            className="px-4 py-2 bg-sage-700 text-white rounded-xl"
          >
            Export PNG
          </button>

          <button
            onClick={onClose}
            className="px-4 py-2 bg-red-500 text-white rounded-xl"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
}
