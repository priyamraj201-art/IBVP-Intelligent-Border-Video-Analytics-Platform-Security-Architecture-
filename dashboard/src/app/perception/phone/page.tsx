"use client";

import { useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Camera, RefreshCw, Radio, PhoneOff, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

function PhoneStreamer() {
  const searchParams = useSearchParams();
  const hubId = searchParams.get("hub");

  const [status, setStatus] = useState<"idle" | "requesting" | "connecting" | "streaming" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startStreaming = (ws: WebSocket) => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    
    intervalRef.current = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      if (!videoRef.current || !canvasRef.current) return;
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (video.videoWidth === 0 || video.videoHeight === 0) return;
      
      // Keep canvas resolution low to ensure fast frame encoding (e.g., 640x480 max)
      const scale = Math.min(1.0, 640 / video.videoWidth);
      canvas.width = video.videoWidth * scale;
      canvas.height = video.videoHeight * scale;
      
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // Extract base64 jpeg
        const frameData = canvas.toDataURL("image/jpeg", 0.7);
        ws.send(frameData);
      }
    }, 100); // Send at ~10 FPS
  };

  const initCameraAndStream = async (mode: "environment" | "user") => {
    try {
      setStatus("requesting");
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: mode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      if (!hubId) {
        setStatus("error");
        setErrorMsg("No Hub ID provided in URL.");
        return;
      }

      setStatus("connecting");
      
      // Connect to the backend WebSocket using dynamic hostname
      const backendIp = window.location.hostname;
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${backendIp}:8000/ws/camera/${hubId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log("Connected to AI Backend");
        setStatus("streaming");
        startStreaming(ws);
      };
      
      ws.onclose = () => {
        setStatus("error");
        setErrorMsg("Disconnected from AI Backend");
        if (intervalRef.current) clearInterval(intervalRef.current);
      };
      
      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        setStatus("error");
        setErrorMsg("Connection to AI Backend failed. Ensure server.py is running.");
      };

    } catch (err: any) {
      console.error(err);
      setStatus("error");
      setErrorMsg(err.message || "Could not access camera");
    }
  };

  useEffect(() => {
    if (hubId) {
      initCameraAndStream(facingMode);
    } else {
      setStatus("error");
      setErrorMsg("Missing ?hub= param in URL");
    }
    
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []); 

  const toggleCamera = () => {
    const newMode = facingMode === "environment" ? "user" : "environment";
    setFacingMode(newMode);
    initCameraAndStream(newMode);
  };

  const disconnect = () => {
    if (wsRef.current) wsRef.current.close();
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStatus("idle");
  };

  return (
    <div className="fixed inset-0 bg-black flex flex-col text-white">
      {/* Hidden canvas for frame extraction */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Video Background */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${facingMode === "user" ? "scale-x-[-1]" : ""} ${status === "streaming" ? "opacity-100" : "opacity-30"}`}
      />
      
      {/* Overlay UI */}
      <div className="relative z-10 flex flex-col h-full bg-gradient-to-b from-black/80 via-transparent to-black/90 pb-8 pt-12 px-6">
        
        {/* Header */}
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-xl font-bold tracking-tight">Camera Node</h1>
            <p className="text-sm text-gray-400 font-mono mt-1">NODE: {hubId || "NONE"}</p>
          </div>
          
          <div className="flex flex-col items-end">
            {status === "streaming" && (
              <div className="flex items-center gap-2 px-3 py-1 bg-green-500/20 border border-green-500/50 rounded-full text-green-400 text-xs font-medium">
                <Radio className="w-3 h-3 animate-pulse" />
                LIVE (AI)
              </div>
            )}
            {status === "connecting" && (
              <div className="flex items-center gap-2 px-3 py-1 bg-yellow-500/20 border border-yellow-500/50 rounded-full text-yellow-400 text-xs font-medium">
                <RefreshCw className="w-3 h-3 animate-spin" />
                CONNECTING
              </div>
            )}
          </div>
        </div>
        
        {/* Status Messages */}
        <div className="flex-1 flex items-center justify-center pointer-events-none">
          {status === "error" && (
            <div className="bg-destructive/90 border border-destructive px-6 py-4 rounded-xl flex flex-col items-center text-center max-w-sm">
              <AlertTriangle className="w-8 h-8 mb-2" />
              <h2 className="font-bold mb-1">Connection Error</h2>
              <p className="text-sm opacity-90">{errorMsg}</p>
              <Button onClick={() => initCameraAndStream(facingMode)} variant="outline" className="mt-4 bg-white/10 border-white/20 hover:bg-white/20 text-white pointer-events-auto">
                Retry Connection
              </Button>
            </div>
          )}
          {status === "requesting" && (
            <div className="flex flex-col items-center">
              <Camera className="w-10 h-10 mb-4 animate-pulse opacity-50" />
              <p className="font-medium">Requesting camera access...</p>
            </div>
          )}
        </div>
        
        {/* Controls */}
        <div className="flex justify-around items-center pt-6 border-t border-white/10">
          <Button
            variant="ghost" 
            size="icon"
            onClick={toggleCamera}
            disabled={status !== "streaming" && status !== "connecting"}
            className="w-14 h-14 rounded-full bg-white/10 hover:bg-white/20 text-white disabled:opacity-50"
          >
            <RefreshCw className="w-6 h-6" />
          </Button>
          
          <Button
            variant="destructive"
            size="icon"
            onClick={disconnect}
            disabled={status === "idle"}
            className="w-20 h-20 rounded-full shadow-[0_0_20px_rgba(239,68,68,0.4)]"
          >
            <PhoneOff className="w-8 h-8" />
          </Button>
        </div>
        
      </div>
    </div>
  );
}

export default function PhonePage() {
  return (
    <Suspense fallback={<div className="fixed inset-0 bg-black flex items-center justify-center text-white">Loading...</div>}>
      <PhoneStreamer />
    </Suspense>
  );
}
