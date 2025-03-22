import {
    ChevronLeft, ChevronRight, ChevronUp, RefreshCw, Monitor,
    Search, Grid, List, MoreHorizontal, Folder, Clock,
    File, HardDrive, Image, Music, Video, Download, Sun,
    Home, Globe, Camera, Code, Settings, Star, Cloud
} from 'lucide-react';

// Title Bar Component
const TitleBar = () => {
    return (
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

    );
};

export default TitleBar;