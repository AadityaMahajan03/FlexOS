import React, { useState, useEffect } from 'react';
import {
    ChevronLeft, ChevronRight, ChevronUp, RefreshCw, Monitor,
    Search, Grid, List, MoreHorizontal, Folder, Clock,
    File, HardDrive, Image, Music, Video, Download, Sun,
    Home, Globe, Camera, Code, Settings, Star, Cloud
} from 'lucide-react';

const WindowsExplorer = ({ onNavigate }) => {
    const [viewMode, setViewMode] = useState('grid');
    const [selectedItem, setSelectedItem] = useState(null);
    const [isSearchFocused, setIsSearchFocused] = useState(false);
    const [currentTime, setCurrentTime] = useState(new Date());
    const [hoveredDrive, setHoveredDrive] = useState(null);
    const [isScrolled, setIsScrolled] = useState(false);

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const handleScroll = (e) => {
        setIsScrolled(e.target.scrollTop > 20);
    };

    const drives = [
        {
            name: 'OS (C:)',
            used: 236,
            total: 387,
            icon: <HardDrive className="w-6 h-6" />,
            color: 'from-cyan-500 to-blue-600',
            accent: 'blue',
            glow: 'cyan'
        },
        {
            name: 'DATA (D:)',
            used: 610,
            total: 931,
            icon: <HardDrive className="w-6 h-6" />,
            color: 'from-fuchsia-500 to-purple-600',
            accent: 'purple',
            glow: 'fuchsia'
        }
    ];

    const sidebarItems = [
        { name: 'Home', icon: <Home className="w-5 h-5" />, color: 'text-cyan-400' },
        { name: 'Favorites', icon: <Star className="w-5 h-5" />, color: 'text-amber-400' },
        { name: 'Documents', icon: <File className="w-5 h-5" />, color: 'text-emerald-400' },
        { name: 'Desktop', icon: <Monitor className="w-5 h-5" />, color: 'text-fuchsia-400' },
        { name: 'Downloads', icon: <Download className="w-5 h-5" />, color: 'text-blue-400' },
        { name: 'Pictures', icon: <Image className="w-5 h-5" />, color: 'text-rose-400' },
        { name: 'Music', icon: <Music className="w-5 h-5" />, color: 'text-red-400' },
        { name: 'Videos', icon: <Video className="w-5 h-5" />, color: 'text-orange-400' },
        { name: 'Cloud', icon: <Cloud className="w-5 h-5" />, color: 'text-sky-400' }
    ];

    return (
        <div className="h-screen bg-[#0A0A0F] bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white overflow-hidden">
            {/* Enhanced Glassmorphic Title Bar */}
            <div className="h-12 bg-black/40 backdrop-blur-2xl flex items-center justify-between px-6 border-b border-white/10 relative z-50">
                <div className="flex items-center gap-4">
                    <div className="relative group">
                        <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg blur opacity-0 group-hover:opacity-75 transition duration-1000 group-hover:duration-200" />
                        <Folder className="relative w-6 h-6 text-cyan-400 transform group-hover:scale-110 transition-all duration-300" />
                    </div>
                    <span className="text-sm font-medium bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500">flexos</span>
                </div>
                <div className="flex gap-3">
                    {['−', '□', '×'].map((btn, i) => (
                        <button
                            key={i}
                            className={`w-8 h-8 flex items-center justify-center rounded-lg transition-all duration-300 bg-white/5
                                ${btn === '×' ? 'hover:bg-red-500/20 hover:text-red-400' : 'hover:bg-white/10'}`}
                        >
                            <span className="text-gray-400 hover:text-white transition-colors">{btn}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Enhanced Navigation Bar with Blur Effect */}
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


            {/* Enhanced Status Bar */}
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

            {/* Main Content Area with Enhanced Scroll */}
            <div className="flex h-[calc(100vh-124px)]">
                {/* Enhanced Sidebar with Hover Effects */}
                <div className="w-72 bg-black/40 backdrop-blur-2xl p-4 border-r border-white/10">
                    <div className="space-y-1">
                        {sidebarItems.map((item, index) => (
                            <button
                                key={index}
                                className={`w-full flex items-center gap-4 px-4 py-3 rounded-xl text-sm transition-all duration-300 relative overflow-hidden
                                    ${selectedItem === item.name
                                        ? 'bg-white/10 border border-white/10'
                                        : 'hover:bg-white/5 border border-transparent'}`}
                                onClick={() => setSelectedItem(item.name)}
                            >
                                <div className={`${item.color} transition-colors duration-300 relative z-10`}>
                                    {item.icon}
                                </div>
                                <span className="font-medium relative z-10">{item.name}</span>
                                {selectedItem === item.name && (
                                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 animate-pulse" />
                                )}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Enhanced Content Area with Scroll Detection */}
                <div className="flex-1 p-6 overflow-auto" onScroll={handleScroll}>
                    {/* Enhanced Toolbar */}
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex gap-3">
                            {['grid', 'list'].map((mode) => (
                                <button
                                    key={mode}
                                    className={`w-10 h-10 flex items-center justify-center rounded-xl transition-all duration-300 relative overflow-hidden
                                        ${viewMode === mode
                                            ? 'bg-cyan-500/20 text-cyan-400 ring-2 ring-cyan-400/20'
                                            : 'hover:bg-white/10'}`}
                                    onClick={() => setViewMode(mode)}
                                >
                                    {mode === 'grid' ? <Grid className="w-4 h-4" /> : <List className="w-4 h-4" />}
                                </button>
                            ))}
                        </div>
                        <button className="w-10 h-10 flex items-center justify-center rounded-xl hover:bg-white/10 transition-all duration-300">
                            <Settings className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Enhanced Drive Cards with Glow Effects */}
                    <div className="grid grid-cols-2 gap-6">
                        {drives.map((drive, index) => (
                            <div
                                key={index}
                                className="group relative bg-black/40 backdrop-blur-2xl rounded-2xl p-6 border border-white/10 
                                    hover:border-cyan-500/50 transition-all duration-500 cursor-pointer overflow-hidden"
                                onMouseEnter={() => setHoveredDrive(index)}
                                onMouseLeave={() => setHoveredDrive(null)}
                            >
                                <div className="relative z-10">
                                    <div className="flex items-center gap-4 mb-6">
                                        <div className={`p-3 rounded-xl bg-gradient-to-br ${drive.color} 
                                            transform group-hover:scale-110 transition-all duration-500 relative`}>
                                            <div className={`absolute inset-0 bg-${drive.glow}-500/50 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                                            {drive.icon}
                                        </div>
                                        <span className="text-lg font-medium group-hover:text-cyan-400 transition-colors duration-300">{drive.name}</span>
                                    </div>
                                    <div className="w-full h-3 bg-white/5 rounded-full overflow-hidden mb-4">
                                        <div
                                            className={`h-full bg-gradient-to-r ${drive.color} relative
                                                transform transition-all duration-700 ease-out group-hover:scale-105`}
                                            style={{ width: `${(drive.used / drive.total) * 100}%` }}
                                        >
                                            <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                                        </div>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm text-gray-400 group-hover:text-white/80 transition-colors duration-300">
                                            {drive.used} GB free of {drive.total} GB
                                        </span>
                                        <span className="text-sm font-medium text-white/70 group-hover:text-cyan-400 transition-colors duration-300">
                                            {Math.round((drive.used / drive.total) * 100)}%
                                        </span>
                                    </div>
                                </div>
                                {/* Enhanced animated background gradient */}
                                <div className={`absolute inset-0 bg-gradient-to-br ${drive.color} opacity-0 
                                    group-hover:opacity-10 transition-opacity duration-500 blur-xl`} />
                            </div>
                        ))}
                    </div>
                </div>
            </div>

        </div>
    );
};

export default WindowsExplorer;