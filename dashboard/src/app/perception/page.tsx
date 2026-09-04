"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Network, QrCode, Copy, Check, Camera, RefreshCw } from "lucide-react";
import QRCode from "react-qr-code";

export default function SharedPerceptionPage() {
  const [hubId, setHubId] = useState<string>("");
  const [nodes, setNodes] = useState<string[]>([]);
  const [origin, setOrigin] = useState<string>("");
  const [copied, setCopied] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"connecting" | "online" | "error">("connecting");

  useEffect(() => {
    setOrigin(window.location.origin);
    
    // Generate a random 6-character alphanumeric ID for the next node connection
    setHubId(Math.random().toString(36).substring(2, 8).toUpperCase());

    // Poll the backend for active nodes
    const fetchNodes = async () => {
      try {
        const backendIp = window.location.hostname;
        const res = await fetch(`http://${backendIp}:8000/api/nodes`);
        if (res.ok) {
          const activeNodes = await res.json();
          setNodes(activeNodes);
          setBackendStatus("online");
        } else {
          setBackendStatus("error");
        }
      } catch (err) {
        setBackendStatus("error");
      }
    };

    fetchNodes();
    const intervalId = setInterval(fetchNodes, 2000); // Poll every 2 seconds

    return () => {
      clearInterval(intervalId);
    };
  }, []);

  const phoneUrl = `${origin}/perception/phone?hub=${hubId}`;

  const copyLink = () => {
    navigator.clipboard.writeText(phoneUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex h-full flex-col p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Shared Perception</h1>
          <p className="text-muted-foreground mt-1">
            AI-Processed Multi-Node Video Streams.
          </p>
        </div>
        <div className="flex gap-4">
          <Badge 
            variant="outline" 
            className={`px-4 py-1.5 border ${backendStatus === "online" ? "border-green-500 text-green-500 bg-green-500/10" : "border-destructive text-destructive bg-destructive/10"}`}
          >
            {backendStatus === "online" ? "BACKEND ONLINE" : backendStatus === "error" ? "BACKEND OFFLINE" : "CONNECTING..."}
          </Badge>
          <Badge 
            variant="outline" 
            className={`px-4 py-1.5 border ${nodes.length > 0 ? "border-primary text-primary shadow-[0_0_10px_rgba(34,211,238,0.2)] bg-primary/10" : "border-muted text-muted-foreground"}`}
          >
            <Network className={`w-4 h-4 mr-2 ${nodes.length > 0 ? "animate-spin-slow" : ""}`} />
            {nodes.length} {nodes.length === 1 ? "NODE" : "NODES"} CONNECTED
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1 min-h-0">
        
        {/* Left Panel: Connect Info */}
        <Card className="col-span-1 border-border/50 bg-secondary/20 flex flex-col items-center p-8 relative overflow-hidden">
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />
          
          <div className="relative z-10 flex flex-col items-center text-center w-full">
            <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mb-6 border border-primary/30">
              <QrCode className="w-8 h-8 text-primary" />
            </div>
            
            <h3 className="font-semibold text-lg mb-2">Connect New Node</h3>
            <p className="text-sm text-muted-foreground mb-8">
              Scan this QR code to add a new AI camera node to the perception network.
            </p>

            <div className="bg-white p-4 rounded-xl shadow-lg mb-8">
              {hubId ? (
                <QRCode value={phoneUrl} size={160} />
              ) : (
                <div className="w-[160px] h-[160px] flex items-center justify-center bg-gray-100 text-gray-400">
                  <span className="animate-pulse">Generating...</span>
                </div>
              )}
            </div>

            <div className="w-full">
              <label className="text-xs text-muted-foreground uppercase tracking-wider mb-2 block text-left">
                New Node ID
              </label>
              <div className="flex items-center gap-2">
                <div className="bg-black/50 border border-border/50 rounded-md px-3 py-2 flex-1 font-mono text-center text-lg tracking-widest text-primary">
                  {hubId || "..."}
                </div>
                <button 
                  onClick={copyLink}
                  className="p-2.5 bg-primary/10 hover:bg-primary/20 border border-primary/30 rounded-md text-primary transition-colors"
                  title="Copy Node Link"
                >
                  {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                </button>
              </div>
              <Button 
                variant="outline" 
                className="w-full mt-4 border-primary/30 text-primary hover:bg-primary/10"
                onClick={() => setHubId(Math.random().toString(36).substring(2, 8).toUpperCase())}
              >
                <RefreshCw className="w-4 h-4 mr-2" /> Generate New ID
              </Button>
            </div>
          </div>
        </Card>

        {/* Right Panel: Video Grid */}
        <Card className="col-span-1 lg:col-span-3 border-border/50 bg-black/40 p-6 flex flex-col relative overflow-y-auto">
          {nodes.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground opacity-60">
              <Camera className="w-16 h-16 mb-4 opacity-50" strokeWidth={1} />
              <p className="text-lg">Waiting for active AI nodes...</p>
              <p className="text-sm mt-2">Connect a mobile node or start the local camera.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 auto-rows-max">
              {nodes.map((nodeId) => (
                <div key={nodeId} className="relative rounded-lg overflow-hidden border border-border/50 bg-black shadow-lg flex items-center justify-center aspect-video group">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={`http://${origin ? new URL(origin).hostname : 'localhost'}:8000/api/stream/${nodeId}`}
                    alt={`AI Stream for ${nodeId}`}
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-3 left-3 bg-black/70 backdrop-blur-sm px-2.5 py-1 rounded-md text-xs font-mono border border-primary/30 flex items-center gap-2 text-primary shadow-sm">
                    <span className="w-1.5 h-1.5 rounded-full bg-destructive animate-pulse" />
                    {nodeId}
                  </div>
                  
                  {/* Decorative corners */}
                  <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-primary/50 m-2 pointer-events-none" />
                  <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-primary/50 m-2 pointer-events-none" />
                  <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-primary/50 m-2 pointer-events-none" />
                  <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-primary/50 m-2 pointer-events-none" />
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
