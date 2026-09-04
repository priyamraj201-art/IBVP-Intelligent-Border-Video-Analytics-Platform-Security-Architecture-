"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Network, Eye, Crosshair } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = [
  { time: '00:00', threats: 2 },
  { time: '04:00', threats: 5 },
  { time: '08:00', threats: 12 },
  { time: '12:00', threats: 8 },
  { time: '16:00', threats: 15 },
  { time: '20:00', threats: 7 },
  { time: '24:00', threats: 3 },
];

export default function SharedPerceptionPage() {
  return (
    <div className="flex h-full flex-col p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Shared Perception</h1>
          <p className="text-muted-foreground mt-1">Multi-node synchronization and cross-camera tracking.</p>
        </div>
        <Badge variant="outline" className="px-4 py-1.5 border-primary text-primary shadow-[0_0_10px_rgba(34,211,238,0.2)] bg-primary/10">
          <Network className="w-4 h-4 mr-2 animate-spin-slow" />
          SYNCED: 24 NODES
        </Badge>
      </div>

      <div className="grid grid-cols-3 gap-6 flex-1">
        
        {/* Tactical Map / Node Graph */}
        <Card className="col-span-2 relative overflow-hidden border-border/50 bg-black flex items-center justify-center p-6 shadow-[0_0_30px_rgba(0,0,0,0.5)]">
           <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />
           
           <div className="relative w-full h-full max-w-2xl max-h-[600px] border border-primary/20 rounded-full flex items-center justify-center">
              {/* Radar Sweep Effect */}
              <div className="absolute inset-0 rounded-full bg-[conic-gradient(from_90deg,transparent_0,transparent_270deg,rgba(34,211,238,0.2)_360deg)] animate-spin" style={{ animationDuration: '4s' }} />
              
              <div className="absolute w-[70%] h-[70%] border border-primary/10 rounded-full" />
              <div className="absolute w-[40%] h-[40%] border border-primary/10 rounded-full" />

              {/* Node 1 */}
              <div className="absolute top-[20%] left-[30%] w-4 h-4 bg-primary rounded-full shadow-[0_0_15px_rgba(34,211,238,1)] flex items-center justify-center">
                <span className="absolute -top-6 text-xs font-mono text-primary">CAM-01</span>
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              </div>

              {/* Node 2 - Threat */}
              <div className="absolute top-[60%] left-[70%] w-4 h-4 bg-destructive rounded-full shadow-[0_0_15px_rgba(239,68,68,1)] flex items-center justify-center">
                <span className="absolute -top-6 text-xs font-mono text-destructive">CAM-08 [ALERT]</span>
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-destructive opacity-75"></span>
              </div>

              {/* Node 3 */}
              <div className="absolute bottom-[25%] left-[40%] w-4 h-4 bg-primary rounded-full shadow-[0_0_15px_rgba(34,211,238,1)] flex items-center justify-center">
                <span className="absolute -bottom-6 text-xs font-mono text-primary">CAM-12</span>
              </div>

              {/* Connection Lines (Simulated with SVG) */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                 <line x1="30%" y1="20%" x2="70%" y2="60%" stroke="rgba(34,211,238,0.2)" strokeWidth="1" strokeDasharray="4 4" />
                 <line x1="70%" y1="60%" x2="40%" y2="75%" stroke="rgba(239,68,68,0.4)" strokeWidth="2" strokeDasharray="4 4" />
                 <line x1="40%" y1="75%" x2="30%" y2="20%" stroke="rgba(34,211,238,0.2)" strokeWidth="1" strokeDasharray="4 4" />
              </svg>

              <Crosshair className="text-primary/30 w-full h-full p-12" strokeWidth={0.5} />
           </div>
        </Card>

        {/* Stats Panel */}
        <div className="col-span-1 flex flex-col gap-6">
          <Card className="p-6 border-border/50 bg-secondary/20">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <Eye className="w-5 h-5 text-primary" />
              Aggregated Threat Level
            </h3>
            <div className="text-5xl font-bold text-destructive mb-2 tracking-tighter">
              ELEVATED
            </div>
            <p className="text-sm text-muted-foreground">Multiple anomalous activities detected across sector boundaries in the past 4 hours.</p>
          </Card>

          <Card className="p-6 border-border/50 bg-secondary/20 flex-1 flex flex-col">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              Cross-Camera Detections (24h)
            </h3>
            <div className="flex-1 w-full min-h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" fontSize={12} />
                  <YAxis stroke="rgba(255,255,255,0.5)" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(34,211,238,0.2)' }}
                    itemStyle={{ color: '#22d3ee' }}
                  />
                  <Line type="monotone" dataKey="threats" stroke="#22d3ee" strokeWidth={2} dot={{ r: 4, fill: '#22d3ee' }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
        
      </div>
    </div>
  );
}
