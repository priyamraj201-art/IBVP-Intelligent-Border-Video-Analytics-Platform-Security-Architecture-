"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ShieldCheck, Lock, User, AlertCircle } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const userStr = localStorage.getItem("trackrz_user");
    if (userStr) {
      router.push("/dashboard");
    }
  }, [router]);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    setTimeout(() => {
      if (username === "Prince" && password === "Prince") {
        localStorage.setItem("trackrz_user", JSON.stringify({ username: "Prince", role: "superadmin" }));
        router.push("/dashboard");
      } else if (username && password) {
        // Guest login
        localStorage.setItem("trackrz_user", JSON.stringify({ username, role: "guest" }));
        router.push("/dashboard");
      } else {
        setError("Please enter valid credentials.");
      }
      setLoading(false);
    }, 800);
  };

  return (
    <div className="relative flex h-screen w-full items-center justify-center overflow-hidden bg-background">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[100px] animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-[30rem] h-[30rem] bg-chart-4/10 rounded-full blur-[120px] animate-pulse delay-1000" />
      </div>

      {/* Top right Theme Toggle */}
      <div className="absolute top-6 right-6 z-20">
        <ThemeToggle />
      </div>

      <div className="z-10 w-full max-w-md px-4">
        <Card className="glass-panel border-primary/20 shadow-2xl p-8 relative overflow-hidden bg-card/40 backdrop-blur-xl">
          {/* Top accent line */}
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary to-transparent" />
          
          <div className="text-center mb-8">
            <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4 border border-primary/20 shadow-[0_0_15px_rgba(var(--primary),0.2)]">
              <ShieldCheck className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">
              BORDER<span className="text-primary">GUARD</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-2">Tactical Surveillance & Intelligence Node</p>
          </div>

          {error && (
            <div className="mb-6 p-3 rounded bg-destructive/10 border border-destructive/20 flex items-center gap-2 text-destructive text-sm">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {error}
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
            <div className="space-y-2">
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input 
                  placeholder="Operator ID / Username" 
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="pl-10 bg-background/50 border-border/50 focus:border-primary/50 transition-colors"
                  required
                />
              </div>
            </div>
            <div className="space-y-2">
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input 
                  type="password"
                  placeholder="Passcode" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10 bg-background/50 border-border/50 focus:border-primary/50 transition-colors"
                  required
                />
              </div>
            </div>
            <Button 
              type="submit" 
              className="w-full relative overflow-hidden group transition-all"
              disabled={loading}
            >
              <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full duration-1000 ease-in-out transition-transform" />
              {loading ? "Authenticating..." : "INITIALIZE UPLINK"}
            </Button>
          </form>

          <div className="mt-8 text-center">
             <p className="text-xs text-muted-foreground/60 font-mono">AUTHORIZED PERSONNEL ONLY. SYSTEM LOGGED.</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
