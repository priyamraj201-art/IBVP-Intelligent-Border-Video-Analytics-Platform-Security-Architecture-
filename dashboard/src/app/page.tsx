"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Video, Target, Car, Users, AlertTriangle, ShieldCheck, UserPlus, UploadCloud } from "lucide-react";
import Link from "next/link";

export default function CameraFeedPage() {
  const [streamError, setStreamError] = useState(false);
  const [eventLogs, setEventLogs] = useState<any[]>([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/alerts");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "FRS_HIT") {
        setEventLogs(prev => {
          // If the last log is the same person, just update its time and a counter
          if (prev.length > 0 && prev[0].msg.includes(data.name)) {
            const updated = [...prev];
            updated[0] = {
               ...updated[0],
               time: new Date(data.timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'}),
               count: (updated[0].count || 1) + 1
            };
            return updated;
          }
          const newLog = {
            id: Date.now() + Math.random(),
            time: new Date(data.timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'}),
            msg: `Identified: ${data.name} (${data.category})`,
            type: data.category === 'VIP' ? 'warning' : (data.category === 'STOLEN' || data.category === 'WANTED' || data.category === 'SUSPECT' ? 'critical' : 'info'),
            icon: Users,
            count: 1
          };
          return [newLog, ...prev].slice(0, 50); // Keep last 50
        });
      }
    };
    return () => ws.close();
  }, []);

  return (
    <div className="flex h-full flex-col p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Camera Feed (Live)</h1>
          <p className="text-muted-foreground mt-1">Real-time surveillance and threat detection node.</p>
        </div>
        <Badge variant="outline" className="px-4 py-1.5 border-primary text-primary shadow-[0_0_10px_rgba(34,211,238,0.2)] bg-primary/10">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse mr-2"></span>
          LIVE
        </Badge>
      </div>

      <div className="flex flex-1 gap-6 overflow-hidden">
        {/* Main Feed Area */}
        <div className="flex flex-col flex-1 gap-4">


          <Card className="flex-1 relative overflow-hidden border-2 border-primary/20 bg-black flex items-center justify-center shadow-lg shadow-primary/5">
            {/* Video Stream */}
            {!streamError ? (
              <img 
                src="http://localhost:8000/api/stream" 
                alt="Live Camera Feed"
                className="absolute inset-0 w-full h-full object-cover"
                onError={() => setStreamError(true)}
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10">
                <AlertTriangle className="w-12 h-12 text-destructive mb-4 animate-pulse" />
                <p className="text-destructive font-mono text-xl tracking-[0.2em]">[ SIGNAL LOST ]</p>
                <p className="text-muted-foreground font-mono text-sm mt-2">CHECK UPLINK CONNECTION</p>
              </div>
            )}

            {/* Tactical Grid Overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none z-10" />
            




          </Card>
        </div>

        {/* Right Sidebar Column */}
        <div className="w-80 flex flex-col gap-4">
          {/* Live Event Log */}
          <Card className="flex-1 flex flex-col min-h-0 border-2 border-border/60 bg-card shadow-lg shadow-black/5">
            <div className="p-4 border-b border-border/50">
              <h3 className="font-semibold flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-primary" />
                Live Event Log
              </h3>
            </div>
            <ScrollArea className="flex-1 p-4">
              <div className="space-y-4">
                {eventLogs.map((log) => (
                  <div key={log.id} className="flex gap-3 text-sm">
                    <div className="mt-0.5">
                      <log.icon className={`w-4 h-4 ${log.type === 'critical' ? 'text-destructive' : log.type === 'warning' ? 'text-yellow-500' : 'text-primary'}`} />
                    </div>
                    <div>
                      <p className="text-muted-foreground text-xs font-mono mb-1">{log.time}</p>
                      <p className={`${log.type === 'critical' ? 'text-destructive font-medium' : 'text-foreground'}`}>
                        {log.msg} {log.count > 1 && <span className="text-primary/70 ml-2">x{log.count}</span>}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </Card>

          {/* Register Identity Card */}
          <Card className="shrink-0 relative overflow-hidden border-2 border-border/60 bg-gradient-to-br from-card to-card/50 shadow-lg shadow-black/5 group cursor-pointer transition-all hover:border-primary/50 hover:shadow-[0_0_30px_rgba(var(--primary),0.15)]">
            <Link href="/register" className="flex flex-col p-6 items-center text-center gap-3 relative z-10 block">
              <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center text-primary group-hover:scale-110 group-hover:bg-primary group-hover:text-primary-foreground transition-all duration-500 shadow-inner">
                <UserPlus className="w-7 h-7" />
              </div>
              <div>
                <h3 className="font-bold text-lg text-foreground tracking-tight">Register Identity</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Enroll a new subject into the biometric database
                </p>
              </div>
            </Link>
            {/* Background Accents */}
            <div className="absolute top-0 right-0 -mt-6 -mr-6 w-32 h-32 bg-primary/10 rounded-full blur-2xl group-hover:bg-primary/20 transition-all duration-500 pointer-events-none" />
            <div className="absolute bottom-0 left-0 -mb-6 -ml-6 w-24 h-24 bg-primary/5 rounded-full blur-xl group-hover:bg-primary/10 transition-all duration-500 pointer-events-none" />
          </Card>
        </div>
      </div>
    </div>
  );
}
