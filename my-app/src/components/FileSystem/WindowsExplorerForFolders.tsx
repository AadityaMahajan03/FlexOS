import React, { useState, useEffect } from 'react';
import {
    ChevronLeft, ChevronRight, ChevronUp, RefreshCw, Monitor,
    Search, Grid, List, MoreHorizontal, Folder, Clock,
    File, HardDrive, Image, Music, Video, Download, Sun,
    Home, Globe, Camera, Code, Settings, Star, Cloud
} from 'lucide-react';
import MainContent2 from './MainContent2';
import TitleBar from './TitleBar';
import NavigationBar from './NavigationBar';
import StatusBar from './StatusBar';

const WindowsExplorerForFolders = ({ onNavigate, filename }) => {
    const [isSearchFocused, setIsSearchFocused] = useState(false);
    const [currentTime, setCurrentTime] = useState(new Date());
    const [isScrolled, setIsScrolled] = useState(false);
    const [drives, setDrives] = useState([]);
    const [root, setRoot] = useState([]);
    useEffect(() => {
        fetch(`http://127.0.0.1:5000/partition/${filename}`)
            .then((response) => response.json())
            .then((data) => {
                console.log("Fetched drives:", data.directories);
                setDrives(data.directories);  // Ensure 'folders' is correct
                setRoot(data.root);
                console.log("Root ", root);
            })
            .catch((error) => console.error("Error fetching partitions:", error));
    }, []);

    useEffect(() => {
        console.log("Updated Root:", root);
    }, [root]);  // Runs whenever `root` changes

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    const handleScroll = (e) => {
        setIsScrolled(e.target.scrollTop > 20);
    };


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
            <TitleBar />

            {/* Enhanced Navigation Bar with Blur Effect */}
            <NavigationBar />

            {/* Enhanced Status Bar */}
            <StatusBar />

            {/* Main Content Area with Enhanced Scroll */}
            <MainContent2 sidebarItems={sidebarItems} drives={drives} handleScroll={handleScroll} onNavigate={onNavigate} root={root} />

        </div>
    );
};

export default WindowsExplorerForFolders;