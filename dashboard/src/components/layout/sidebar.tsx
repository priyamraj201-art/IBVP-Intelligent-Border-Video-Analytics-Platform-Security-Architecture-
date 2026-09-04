"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Video, Network, Map as MapIcon, LayoutDashboard, ShieldAlert, Menu } from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/theme-toggle";

const navigation = [
  { name: "Camera Feed", href: "/", icon: Video },
  { name: "Shared Perception", href: "/perception", icon: Network },
  { name: "Geofencing", href: "/geofencing", icon: MapIcon },
  { name: "Admin Dashboard", href: "/admin", icon: LayoutDashboard },
];

export function Sidebar() {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className={cn("flex h-full flex-col border-r bg-sidebar/50 backdrop-blur-xl transition-all duration-300", isCollapsed ? "w-20" : "w-64")}>
      <div className={cn("flex h-16 shrink-0 items-center border-b border-border/50", isCollapsed ? "justify-center px-0" : "justify-between px-4")}>
        {!isCollapsed && (
          <div className="flex items-center gap-2 overflow-hidden">
            <ShieldAlert className="h-6 w-6 text-primary animate-pulse shrink-0" />
            <span className="text-lg font-bold tracking-wider text-foreground whitespace-nowrap">
              BORDER<span className="text-primary">GUARD</span>
            </span>
          </div>
        )}
        <button onClick={() => setIsCollapsed(!isCollapsed)} className="p-2 hover:bg-muted rounded-md shrink-0 text-muted-foreground hover:text-foreground transition-colors">
          <Menu className="w-5 h-5" />
        </button>
      </div>
      
      <div className="flex flex-1 flex-col overflow-y-auto pt-6 pb-4 overflow-x-hidden">
        <nav className="flex-1 space-y-2 px-3">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "group flex items-center rounded-md px-3 py-2.5 text-sm font-semibold transition-all duration-200",
                  isCollapsed ? "justify-center" : "gap-x-3",
                  isActive
                    ? "bg-primary/10 text-primary border border-primary/20 shadow-[0_0_15px_rgba(var(--primary),0.1)]"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                )}
                title={isCollapsed ? item.name : undefined}
              >
                <item.icon
                  className={cn(
                    "h-5 w-5 shrink-0 transition-colors",
                    isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                  )}
                  aria-hidden="true"
                />
                {!isCollapsed && <span className="whitespace-nowrap">{item.name}</span>}
              </Link>
            );
          })}
        </nav>
        
        <div className="mt-auto px-3 pb-4 space-y-4">
          <div className={cn("flex items-center justify-center", !isCollapsed && "justify-start px-2")}>
            <ThemeToggle />
          </div>
          {!isCollapsed && (
            <div className="rounded-lg bg-secondary/30 p-4 border border-border/50 backdrop-blur-md">
              <h3 className="text-xs font-semibold text-primary uppercase tracking-widest mb-2 flex items-center gap-2 whitespace-nowrap">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse shrink-0"></span>
                System Status
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                All sectors secure. Multi-spectral sensors online. AI inference node optimal.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
