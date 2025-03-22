// Navigation Bar Component
import React, { useState, useEffect } from 'react';
import {
    ChevronLeft, ChevronRight, ChevronUp, RefreshCw, Monitor,
    Search, Grid, List, MoreHorizontal, Folder, Clock,
    File, HardDrive, Image, Music, Video, Download, Sun,
    Home, Globe, Camera, Code, Settings, Star, Cloud
} from 'lucide-react';

const NavigationBar = () => {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isSearchFocused, setIsSearchFocused] = useState(false);

    return (
        <div className={`h-16 transition-all duration-500 ${isScrolled ? 'bg-black/60' : 'bg-black/40'} backdrop-blur-2xl flex items-center px-6 gap-6 border-b border-white/10 sticky top-0 z-40`}>
            <div className="flex gap-2">
                {[<ChevronLeft />, <ChevronRight />, <ChevronUp />, <RefreshCw />].map((icon, i) => (
                    <button
                        key={i}
                        className="w-10 h-10 flex items-center justify-center rounded-xl hover:bg-white/10 transition-all duration-300 group relative overflow-hidden"
                    >
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-purple-500 opacity-0 group-hover:opacity-20 transition-opacity duration-300" />
                        {React.cloneElement(icon, {
                            className: "relative w-4 h-4 group-hover:text-cyan-400 transition-colors duration-300"
                        })}
                    </button>
                ))}
            </div>

            <div className="flex-1 flex items-center bg-white/5 rounded-xl px-4 py-2.5 border border-white/10 group hover:border-cyan-500/50 transition-all duration-500">
                <div className="flex items-center gap-2 text-sm text-gray-400">
                    <span className="text-white/70">This PC</span>
                    <ChevronRight className="w-4 h-4 text-gray-600" />
                    <span className="text-white/70">DATA (D:)</span>
                    <ChevronRight className="w-4 h-4 text-gray-600" />
                    <span className="text-white group-hover:text-cyan-400 transition-colors duration-300">my-app</span>
                </div>
            </div>

            <div className={`relative flex items-center backdrop-blur-xl rounded-xl px-4 py-2.5 transition-all duration-500
                    ${isSearchFocused
                    ? 'bg-white/10 border-cyan-500/50 ring-4 ring-cyan-500/20'
                    : 'bg-white/5 border border-white/10 hover:border-white/20'}`}>
                <Search className={`w-4 h-4 transition-colors duration-300 ${isSearchFocused ? 'text-cyan-400' : 'text-gray-400'}`} />
                <input
                    type="text"
                    placeholder="Search my-app"
                    className="bg-transparent border-none outline-none px-3 text-sm w-48 placeholder-gray-500"
                    onFocus={() => setIsSearchFocused(true)}
                    onBlur={() => setIsSearchFocused(false)}
                />
            </div>
        </div>
    );
};

export default NavigationBar;