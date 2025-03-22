import React, { useState, useEffect } from 'react';
import {
    ChevronLeft, ChevronRight, ChevronUp, RefreshCw, Monitor,
    Search, Grid, List, MoreHorizontal, Folder, Clock,
    File, HardDrive, Image, Music, Video, Download, Sun,
    Home, Globe, Camera, Code, Settings, Star, Cloud
} from 'lucide-react';



const MainContent3 = ({ sidebarItems, drives, handleScroll, onNavigate }) => {
    const [viewMode, setViewMode] = useState('grid');
    const [selectedItem, setSelectedItem] = useState(null);
    const [isSearchFocused, setIsSearchFocused] = useState(false);
    const [currentTime, setCurrentTime] = useState(new Date());
    const [hoveredDrive, setHoveredDrive] = useState(null);
    const [isScrolled, setIsScrolled] = useState(false);


    const fetchDirectories = async (partitionName) => {
        const encodedPartitionName = encodeURIComponent(partitionName);
        const encodedRoot = encodeURIComponent(root);
        const modifiedPartitionName = `WindowsExplorerForAllRest_${encodedRoot}_${encodedPartitionName}`;
        console.log(modifiedPartitionName);
        onNavigate(modifiedPartitionName);
    };

    const displayAttributes =
    {
        used: 236,
        total: 387,
        icon: <HardDrive className="w-6 h-6" />,
        color: 'from-cyan-500 to-blue-600',
        accent: 'blue',
        glow: 'cyan'
    }
    return (
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
                            onClick={() => fetchDirectories(drive)}

                        >
                            <div className="relative z-10">
                                <div className="flex items-center gap-4 mb-6">
                                    <div className={`p-3 rounded-xl bg-gradient-to-br ${displayAttributes.color} 
                                            transform group-hover:scale-110 transition-all duration-500 relative`}>
                                        <div className={`absolute inset-0 bg-${displayAttributes.glow}-500/50 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                                        {displayAttributes.icon}
                                    </div>
                                    <span className="text-lg font-medium group-hover:text-cyan-400 transition-colors duration-300">{drive}</span>
                                </div>
                            </div>
                            {/* Enhanced animated background gradient */}
                            <div className={`absolute inset-0 bg-gradient-to-br ${displayAttributes.color} opacity-0 
                                    group-hover:opacity-10 transition-opacity duration-500 blur-xl`} />
                        </div>
                    ))}
                </div>
            </div>
        </div>

    );
};

export default MainContent3;