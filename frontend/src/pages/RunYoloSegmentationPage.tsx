export default function RunYoloPage() {
    // Retrieve results that were stored after segmentation completed.
    // overlayUrl: the full annotated image (overlay of masks + bounding boxes)
    // stickerUrls: list of individual object sticker PNGs
    const overlayUrl = localStorage.getItem("overlayUrl");
    const stickerUrls = JSON.parse(localStorage.getItem("stickerUrls") ?? "[]");

    // If overlayUrl is missing, it means:
    // - Segmentation did not run yet, OR
    // - User refreshed page (localStorage cleared), OR
    // - No objects were detected.
    // In that case, show a simple fallback message.
    if (!overlayUrl) {
        return (
            <div className="min-h-screen flex items-center justify-center">
            No result found. Go back and upload an image.
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-sage-50 text-sage-900 p-10">
            {/* Page Title */}
            <h1 className="text-3xl font-semibold mb-6">Segmentation Result</h1>

            {/* Display the full overlay image */}
            <div className="rounded-2xl bg-white p-4 border border-sage-200 shadow-sm max-w-4xl mx-auto">
                {/* The overlay image shows all detections + masks on top of the original image */}
                <img src={overlayUrl} alt="overlay" className="w-full rounded-xl" />
            </div>

            {/* If stickers exist, show them below the overlay */}
            {stickerUrls.length > 0 && (
                <div className="mt-10 max-w-4xl mx-auto">
                    <h2 className="text-2xl mb-4">Stickers</h2>
                    
                    {/* Display every sticker as a small image in a grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        {stickerUrls.map((s: string, i: number) => (
                            <img
                                key={i}
                                src={s}         // URL to sticker PNG
                                alt={`sticker ${i}`}    
                                className="rounded-xl shadow-sm"
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
