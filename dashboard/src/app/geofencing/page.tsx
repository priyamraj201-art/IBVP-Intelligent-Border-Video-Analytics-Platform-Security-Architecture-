"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Map, Settings, Hexagon, Pencil, Trash2 } from "lucide-react";

export default function GeofencingPage() {
  return (
    <div className="flex h-full flex-col p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Geofencing & Perimeters</h1>
          <p className="text-muted-foreground mt-1">Define boundaries and configure intrusion alerts.</p>
        </div>
        <Badge variant="outline" className="px-4 py-1.5 border-primary text-primary shadow-[0_0_10px_rgba(34,211,238,0.2)] bg-primary/10">
          <Hexagon className="w-4 h-4 mr-2" />
          3 ACTIVE ZONES
        </Badge>
      </div>

      <div className="flex flex-1 gap-6 overflow-hidden">
        
        {/* Map Interface Placeholder */}
        <Card className="flex-1 relative overflow-hidden border-border/50 bg-black flex items-center justify-center shadow-[0_0_30px_rgba(0,0,0,0.5)]">
           <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />
           
           {/* Topography Mock */}
           <div className="absolute inset-0 opacity-20 pointer-events-none">
              <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
                <path d="M0 200 Q 200 150 400 300 T 800 250" fill="none" stroke="#22d3ee" strokeWidth="1"/>
                <path d="M0 250 Q 250 200 450 350 T 800 300" fill="none" stroke="#22d3ee" strokeWidth="0.5"/>
              </svg>
           </div>

           {/* Drawn Polygon Mock */}
           <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <polygon points="300,150 500,180 480,350 250,300" fill="rgba(239,68,68,0.1)" stroke="#ef4444" strokeWidth="2" strokeDasharray="4 4" />
              {/* Polygon Vertices */}
              <circle cx="300" cy="150" r="4" fill="#ef4444" />
              <circle cx="500" cy="180" r="4" fill="#ef4444" />
              <circle cx="480" cy="350" r="4" fill="#ef4444" />
              <circle cx="250" cy="300" r="4" fill="#ef4444" />
           </svg>
           
           <div className="absolute top-4 right-4 bg-background/80 backdrop-blur border border-border/50 rounded-md p-1 flex flex-col gap-1 shadow-lg">
             <Button variant="ghost" size="icon" className="h-8 w-8 text-primary hover:bg-primary/20">
               <Pencil className="w-4 h-4" />
             </Button>
             <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive hover:bg-destructive/20">
               <Trash2 className="w-4 h-4" />
             </Button>
           </div>

           <div className="absolute top-1/2 left-[38%] text-destructive font-mono text-xs bg-black/50 px-2 py-1 rounded border border-destructive/50">
             ZONE_ALPHA (RESTRICTED)
           </div>
        </Card>

        {/* Settings Panel */}
        <Card className="w-96 flex flex-col border-border/50 bg-secondary/20 backdrop-blur-sm p-6 overflow-y-auto">
          <h3 className="font-semibold text-lg mb-6 flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary" />
            Alert Configurations
          </h3>

          <div className="space-y-6">
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Trigger Conditions</h4>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="person-breach" className="flex flex-col gap-1">
                  <span>Person Breach</span>
                  <span className="text-xs text-muted-foreground">Trigger alert when human detected in zone</span>
                </Label>
                <Switch id="person-breach" defaultChecked className="data-[state=checked]:bg-destructive" />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="vehicle-breach" className="flex flex-col gap-1">
                  <span>Vehicle Breach</span>
                  <span className="text-xs text-muted-foreground">Trigger alert when vehicle detected in zone</span>
                </Label>
                <Switch id="vehicle-breach" defaultChecked className="data-[state=checked]:bg-destructive" />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="loitering" className="flex flex-col gap-1">
                  <span>Loitering Detection</span>
                  <span className="text-xs text-muted-foreground">Trigger alert if target stays &gt; N seconds</span>
                </Label>
                <Switch id="loitering" defaultChecked className="data-[state=checked]:bg-primary" />
              </div>
            </div>

            <div className="h-px w-full bg-border/50" />

            <div className="space-y-4">
              <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Parameters</h4>
              
              <div className="space-y-2">
                <Label htmlFor="loiter-time">Loitering Threshold (seconds)</Label>
                <Input id="loiter-time" type="number" defaultValue={30} className="bg-background/50 border-border/50" />
              </div>

              <div className="space-y-2">
                <Label htmlFor="confidence">Minimum AI Confidence (%)</Label>
                <Input id="confidence" type="number" defaultValue={85} className="bg-background/50 border-border/50" />
              </div>
            </div>

            <Button className="w-full bg-primary text-primary-foreground hover:bg-primary/90 mt-4 shadow-[0_0_15px_rgba(34,211,238,0.4)]">
              Save Configuration
            </Button>
          </div>
        </Card>
        
      </div>
    </div>
  );
}
