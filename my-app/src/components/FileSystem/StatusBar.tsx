import React, { useState, useEffect } from 'react';
import {
    ChevronLeft, ChevronRight, ChevronUp, RefreshCw, Monitor,
    Search, Grid, List, MoreHorizontal, Folder, Clock,
    File, HardDrive, Image, Music, Video, Download, Sun,
    Home, Globe, Camera, Code, Settings, Star, Cloud
} from 'lucide-react';

// Status Bar Component
const StatusBar = () => {
    const [currentTime, setCurrentTime] = useState(new Date());

    return (
        <div className="h-12 bg-black/40 backdrop-blur-2xl flex items-center justify-between px-6 text-sm border-t border-white/10">
            <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse shadow-lg shadow-cyan-500/50" />
                    <span className="text-white/70">2 items selected</span>
                </div>
            </div>
            <div className="flex items-center gap-8">
                <div className="flex items-center gap-2">
                    <Sun className="w-4 h-4 text-amber-400" />
                    <span className="text-white/70">34°C</span>
                </div>
                <div className="text-white/70">ENG</div>
                <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-cyan-400" />
                    <span className="text-white/70">
                        {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                </div>
            </div>
        </div>
    );
};
export default StatusBar;