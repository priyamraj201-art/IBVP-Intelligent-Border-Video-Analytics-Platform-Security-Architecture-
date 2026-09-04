"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Camera, UploadCloud, UserPlus, ArrowLeft, RefreshCw, CheckCircle2, Crop as CropIcon } from "lucide-react";
import Link from "next/link";

import ReactCrop, { Crop, PixelCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';

export default function RegisterPage() {
  const [mode, setMode] = useState<"camera" | "upload">("camera");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  
  // Cropping State
  const [crop, setCrop] = useState<Crop>();
  const [completedCrop, setCompletedCrop] = useState<PixelCrop | null>(null);
  const [croppedImageSrc, setCroppedImageSrc] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Form State
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState("VIP");

  const capturePhoto = async () => {
    try {
      // Fetch a high-res raw snapshot from the backend
      const res = await fetch("http://localhost:8000/api/snapshot/raw");
      if (res.ok) {
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        setImageSrc(objectUrl);
        setCroppedImageSrc(null); // Reset crop
        setCrop(undefined);
        setCompletedCrop(null);
      }
    } catch (err) {
      console.error("Failed to capture snapshot from backend", err);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImageSrc(event.target?.result as string);
        setCroppedImageSrc(null); // Reset crop
        setCrop(undefined);
        setCompletedCrop(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const retakePhoto = () => {
    setImageSrc(null);
    setCroppedImageSrc(null);
    setCrop(undefined);
    setCompletedCrop(null);
  };

  // Extract cropped region using HTML5 Canvas
  const getCroppedImg = useCallback(async () => {
    if (!completedCrop || !imgRef.current) return;
    
    const image = imgRef.current;
    const canvas = document.createElement("canvas");
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;
    
    canvas.width = completedCrop.width;
    canvas.height = completedCrop.height;
    const ctx = canvas.getContext("2d");
    
    if (!ctx) return;
    
    ctx.drawImage(
      image,
      completedCrop.x * scaleX,
      completedCrop.y * scaleY,
      completedCrop.width * scaleX,
      completedCrop.height * scaleY,
      0,
      0,
      completedCrop.width,
      completedCrop.height
    );
    
    return new Promise<string>((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error('Canvas is empty'));
          return;
        }
        resolve(URL.createObjectURL(blob));
      }, "image/jpeg", 1);
    });
  }, [completedCrop]);

  const confirmCrop = async () => {
    try {
      const croppedBlobUrl = await getCroppedImg();
      if (croppedBlobUrl) {
        setCroppedImageSrc(croppedBlobUrl);
      }
    } catch (e) {
      console.error("Failed to crop image", e);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const finalImage = croppedImageSrc || imageSrc;
    if (!finalImage || !fullName) return;

    setIsSubmitting(true);
    
    try {
      // Convert imageSrc (dataUrl or objectUrl) to a real File/Blob
      const res = await fetch(finalImage);
      const blob = await res.blob();

      const formData = new FormData();
      formData.append("name", fullName);
      formData.append("category", role);
      formData.append("image", blob, "face.jpg");

      const response = await fetch("http://localhost:8000/api/register_face", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.status === "success") {
        setIsSuccess(true);
      } else {
        alert("Registration failed: " + data.message);
      }
    } catch (error) {
      console.error("Error submitting face:", error);
      alert("Failed to submit the face registration.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex h-full flex-col p-6 space-y-6 max-w-4xl mx-auto w-full">
      <div className="flex items-center gap-4">
          <Link href="/">
            <Button variant="outline" size="icon" type="button">
              <ArrowLeft className="w-4 h-4" />
            </Button>
          </Link>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Register Identity</h1>
          <p className="text-muted-foreground mt-1">Add a new face to the facial recognition database.</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1">
        
        {/* Left Column: Image Capture/Upload/Crop */}
        <Card className="flex flex-col border-2 border-border/60 bg-card shadow-lg shadow-black/5 overflow-hidden">
          <div className="flex border-b border-border/50">
            <button
              type="button"
              onClick={() => { setMode("camera"); setImageSrc(null); setCroppedImageSrc(null); }}
              className={`flex-1 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${mode === "camera" ? "bg-primary/20 text-primary border-b-2 border-primary" : "text-muted-foreground hover:bg-secondary/20"}`}
            >
              <Camera className="w-4 h-4" /> Use Camera
            </button>
            <button
              type="button"
              onClick={() => { setMode("upload"); setImageSrc(null); setCroppedImageSrc(null); }}
              className={`flex-1 py-3 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${mode === "upload" ? "bg-primary/20 text-primary border-b-2 border-primary" : "text-muted-foreground hover:bg-secondary/20"}`}
            >
              <UploadCloud className="w-4 h-4" /> Upload Photo
            </button>
          </div>

          <div className="flex-1 p-4 flex flex-col items-center justify-center bg-black/40 min-h-[400px] relative">
            {croppedImageSrc ? (
              // Final Cropped Preview
              <div className="relative w-full h-full flex flex-col items-center justify-center group">
                <img src={croppedImageSrc} alt="Cropped preview" className="max-h-full max-w-full rounded-md object-contain shadow-2xl border border-border/50" />
                <Button 
                  type="button" 
                  variant="secondary" 
                  className="absolute bottom-4 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity" 
                  onClick={() => setCroppedImageSrc(null)}
                >
                  <RefreshCw className="w-4 h-4 mr-2" /> Redo Crop
                </Button>
              </div>
            ) : imageSrc ? (
              // Cropping View
              <div className="relative w-full h-full flex flex-col items-center justify-center">
                <ReactCrop 
                  crop={crop} 
                  onChange={c => setCrop(c)} 
                  onComplete={c => setCompletedCrop(c)}
                  aspect={1}
                >
                  <img 
                    ref={imgRef}
                    src={imageSrc} 
                    alt="To be cropped" 
                    className="max-h-full max-w-full rounded-md" 
                    onLoad={(e) => {
                      const { width, height } = e.currentTarget;
                      const size = Math.min(width, height) * 0.9;
                      const x = (width - size) / 2;
                      const y = (height - size) / 2;
                      
                      const defaultCrop: PixelCrop = {
                        unit: 'px',
                        x,
                        y,
                        width: size,
                        height: size,
                      };
                      setCrop(defaultCrop);
                      setCompletedCrop(defaultCrop);
                    }}
                  />
                </ReactCrop>
                <div className="absolute bottom-4 flex gap-2">
                  <Button 
                    type="button" 
                    variant="secondary" 
                    className="shadow-lg" 
                    onClick={retakePhoto}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" /> Retake
                  </Button>
                  <Button 
                    type="button" 
                    className="shadow-lg" 
                    onClick={confirmCrop}
                    disabled={!completedCrop?.width || !completedCrop?.height}
                  >
                    <CropIcon className="w-4 h-4 mr-2" /> Confirm Crop
                  </Button>
                </div>
              </div>
            ) : mode === "camera" ? (
              // Live Camera View (Raw Feed from Backend)
              <div className="relative w-full h-full flex flex-col items-center justify-center">
                <img 
                  src="http://localhost:8000/api/stream/raw"
                  alt="Live Raw Feed"
                  className="w-full h-full object-cover rounded-md shadow-2xl border border-border/50"
                />
                <Button 
                  type="button" 
                  size="lg"
                  className="absolute bottom-6 rounded-full w-16 h-16 p-0 shadow-[0_0_20px_rgba(var(--primary),0.5)] border-4 border-background/50 hover:scale-105 transition-transform"
                  onClick={capturePhoto}
                >
                  <Camera className="w-6 h-6" />
                </Button>
              </div>
            ) : (
              // Upload View
              <div 
                className="w-full h-full border-2 border-dashed border-border/50 rounded-lg flex flex-col items-center justify-center gap-4 text-muted-foreground hover:bg-secondary/20 hover:text-foreground hover:border-primary/50 cursor-pointer transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <UploadCloud className="w-12 h-12 mb-2" />
                <p className="text-sm font-medium">Click to browse or drag and drop</p>
                <p className="text-xs opacity-70">JPG, PNG or WEBP (Max. 5MB)</p>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileUpload} 
                  accept="image/*" 
                  className="hidden" 
                />
              </div>
            )}
          </div>
        </Card>

        {/* Right Column: Identity Details */}
        <Card className="flex flex-col border-2 border-border/60 bg-card shadow-lg shadow-black/5">
          <div className="p-6 border-b border-border/50 bg-secondary/10">
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <UserPlus className="w-5 h-5 text-primary" />
              Identity Details
            </h3>
          </div>
          
          <div className="p-6 flex flex-col gap-6 flex-1">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Full Name</label>
              <Input 
                placeholder="e.g. John Doe" 
                className="bg-background h-12 text-lg" 
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Category / Role</label>
              <select 
                className="flex h-12 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-base shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                value={role}
                onChange={(e) => setRole(e.target.value)}
              >
                <option value="VIP">VIP / Whitelist</option>
                <option value="STAFF">Staff Member</option>
                <option value="SUSPECT">Suspect / Watchlist</option>
                <option value="WANTED">Wanted / Critical</option>
                <option value="UNKNOWN">Unknown / Other</option>
              </select>
            </div>

            <div className="flex-1"></div>

            {isSuccess ? (
              <div className="flex flex-col items-center justify-center p-6 bg-green-500/10 border border-green-500/20 rounded-lg text-green-500 gap-3">
                <CheckCircle2 className="w-10 h-10" />
                <p className="font-semibold text-lg">Identity Registered!</p>
                <Button variant="outline" className="mt-2 text-foreground" onClick={() => { setIsSuccess(false); setImageSrc(null); setCroppedImageSrc(null); setFullName(""); }}>
                  Register Another
                </Button>
              </div>
            ) : (
              <Button 
                type="submit" 
                size="lg" 
                disabled={!(croppedImageSrc || imageSrc) || !fullName || isSubmitting}
                className="w-full text-lg h-14 font-semibold shadow-[0_0_15px_rgba(var(--primary),0.3)] transition-all"
              >
                {isSubmitting ? (
                  <span className="flex items-center gap-2">
                    <RefreshCw className="w-5 h-5 animate-spin" /> Processing...
                  </span>
                ) : (
                  "Save Identity to Database"
                )}
              </Button>
            )}
          </div>
        </Card>
      </form>
    </div>
  );
}
