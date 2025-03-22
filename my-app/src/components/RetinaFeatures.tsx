import React, { useState } from 'react';
import { Pause, Hand, Mouse, Layout, Move, ScrollText, ZoomIn, Sparkles } from 'lucide-react';

const RetinaFeatures = ({ onNavigate }) => {
    const [isHeaderHovered, setIsHeaderHovered] = useState(false);
    const [hoveredItem, setHoveredItem] = useState(null);
    const [isHovered, setIsHovered] = useState(false);


    const equipmentItems = [
        {
            name: "Retina Mouse",
            description: "Use eyes as Mouse Control",
            icon: <Hand className="w-16 h-16" strokeWidth={1} />,
            color: "from-cyan-950/40 via-cyan-800/20 to-cyan-950/40",
            glow: "cyan",
            indicators: ['red', 'green', 'yellow'],
            apiEndpoint: "http://127.0.0.1:5000/retinamouse/start"  // API to start the virtual mouse

        }
    ];

    const getGlowColor = (color) => {
        const colors = {
            cyan: 'shadow-[0_0_25px_rgba(6,182,212,0.4)]',
            purple: 'shadow-[0_0_25px_rgba(147,51,234,0.4)]',
            emerald: 'shadow-[0_0_25px_rgba(16,185,129,0.4)]',
            amber: 'shadow-[0_0_25px_rgba(217,119,6,0.4)]',
            red: 'shadow-[0_0_25px_rgba(185,28,28,0.4)]',
            blue: 'shadow-[0_0_25px_rgba(37,99,235,0.4)]'
        };
        return colors[color] || colors.cyan;
    };

    const getTextColor = (glow) => {
        const colors = {
            cyan: 'text-cyan-300 group-hover:text-cyan-200',
            purple: 'text-purple-300 group-hover:text-purple-200',
            emerald: 'text-emerald-300 group-hover:text-emerald-200',
            amber: 'text-amber-300 group-hover:text-amber-200',
            red: 'text-red-300 group-hover:text-red-200',
            blue: 'text-blue-300 group-hover:text-blue-200'
        };
        return colors[glow] || colors.cyan;
    };

    return (
        <div className="min-h-screen bg-[#0a0a0a] flex flex-col items-center p-8 relative overflow-hidden">
            {/* Enhanced Cyberpunk Grid Background */}
            <div className="absolute inset-0">
                <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.05)_1px,transparent_1px)] bg-[size:40px_40px]" />
                <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.02)_2px,transparent_2px),linear-gradient(90deg,rgba(6,182,212,0.02)_2px,transparent_2px)] bg-[size:20px_20px]" />
            </div>
            <button
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
                onClick={() => onNavigate('BiometricDesktopUI')}
                className="group relative flex items-center gap-3 px-6 py-3 bg-transparent overflow-hidden mb-5"
            >
                {/* Background layers */}
                <div className="absolute inset-0 bg-cyan-950/50 border border-cyan-500/20 rounded-lg transform transition-transform duration-300 group-hover:scale-105" />
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/10 to-cyan-500/0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                {/* Animated border */}
                <div className="absolute inset-0 rounded-lg">
                    <div className="absolute inset-0 border border-cyan-500/50 rounded-lg transform transition-all duration-300 group-hover:scale-105 group-hover:border-cyan-400" />
                    <div className="absolute inset-0 border border-cyan-400/20 rounded-lg blur-sm transform transition-all duration-300 group-hover:scale-110 group-hover:border-cyan-300/40" />
                </div>

                {/* Glow effects */}
                <div className="absolute inset-0 bg-cyan-500/5 rounded-lg blur-xl transform transition-all duration-300 group-hover:bg-cyan-400/20 group-hover:scale-110" />

                {/* Content */}
                <div className="relative flex items-center gap-3 ">
                    <Pause className="w-5 h-5  text-cyan-400 transform transition-all duration-300 group-hover:text-cyan-300 group-hover:scale-110" />
                    <span className="text-lg font-bold font-mono tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-cyan-300">
                        Back
                    </span>
                    <Sparkles className="w-4 h-4 text-cyan-400/70 transform transition-all duration-300 group-hover:text-cyan-300 group-hover:rotate-180" />
                </div>

                {/* Scan line effect */}
                <div className="absolute inset-0 overflow-hidden opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent absolute top-0 -left-full animate-[scan_2s_linear_infinite]" />
                </div>
            </button>


            {/* Enhanced Title Section with Hover Effects */}
            <div
                className="text-center mb-16 relative z-10 p-8"
                onMouseEnter={() => setIsHeaderHovered(true)}
                onMouseLeave={() => setIsHeaderHovered(false)}
            >
                {/* Animated Background Effect */}
                <div className={`
          absolute inset-0 bg-gradient-to-r from-cyan-900/0 via-cyan-900/10 to-cyan-900/0
          transition-all duration-700 transform
          ${isHeaderHovered ? 'scale-110 opacity-100' : 'scale-100 opacity-0'}
        `} />

                {/* Animated Border */}
                <div className={`
          absolute inset-0 border border-cyan-500/20
          transition-all duration-700
          ${isHeaderHovered ? 'scale-105 border-cyan-400/40 rounded-2xl' : 'scale-100 rounded-xl'}
        `} />

                {/* Main Title Container */}
                <div className="relative inline-block group">
                    {/* Sparkle Icons */}
                    <Sparkles
                        className={`
              absolute -left-12 -top-8 w-8 h-8 text-cyan-400/50
              transition-all duration-500 transform
              ${isHeaderHovered ? 'scale-125 rotate-180 text-cyan-300' : 'scale-100 rotate-0'}
            `}
                    />
                    <Sparkles
                        className={`
              absolute -right-12 -bottom-8 w-8 h-8 text-cyan-400/50
              transition-all duration-500 transform
              ${isHeaderHovered ? 'scale-125 rotate-180 text-cyan-300' : 'scale-100 rotate-0'}
            `}
                    />

                    {/* Main Title */}
                    <h1 className={`
            text-6xl font-black bg-clip-text text-transparent 
            bg-gradient-to-r from-cyan-400 via-blue-300 to-cyan-400 
            mb-6 font-mono tracking-widest relative
            transition-all duration-500
            ${isHeaderHovered ? 'tracking-[0.2em] scale-105' : ''}
          `}>
                        RETINA CONTROLS

                        {/* Animated Underline */}
                        <div className={`
              absolute bottom-0 left-0 w-full h-1 
              bg-gradient-to-r from-cyan-500 via-blue-400 to-cyan-500
              transform origin-left transition-transform duration-500
              ${isHeaderHovered ? 'scale-x-100' : 'scale-x-0'}
            `} />
                    </h1>

                    {/* Glow Effect */}
                    <div className={`
            absolute -inset-4 bg-cyan-500/20 blur-2xl -z-10
            transition-all duration-700
            ${isHeaderHovered ? 'opacity-100 scale-110' : 'opacity-40 scale-100'}
          `} />
                </div>

                {/* Enhanced Subheading */}
                <div className="relative">
                    <p className={`
            text-lg font-mono tracking-wider font-semibold
            transition-all duration-500 transform
            ${isHeaderHovered ? 'text-cyan-300 scale-105' : 'text-cyan-500'}
          `}>
                        Advanced retina cursor control
                    </p>
                    <p className={`
            text-base font-mono tracking-wider mt-2
            transition-all duration-500
            ${isHeaderHovered ? 'text-cyan-400' : 'text-cyan-600'}
          `}>
                        for seamless interaction control
                    </p>

                    {/* Animated Lines */}
                    <div className="flex justify-center gap-4 mt-4">
                        {[...Array(3)].map((_, i) => (
                            <div
                                key={i}
                                className={`
                  h-0.5 w-12 bg-cyan-500/30
                  transition-all duration-500 transform
                  ${isHeaderHovered ? 'w-16 bg-cyan-400/50' : ''}
                `}
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 w-full max-w-6xl relative z-10">
                {equipmentItems.map((item, index) => (
                    <div
                        key={index}
                        className="relative transform transition-all duration-500 hover:scale-105"
                        onMouseEnter={() => setHoveredItem(index)}
                        onMouseLeave={() => setHoveredItem(null)}
                        onClick={async () => {
                            if (item.apiEndpoint) {
                                try {
                                    const response = await fetch(item.apiEndpoint, { method: "POST" });
                                    const data = await response.json();
                                    alert(`Status: ${data.status}`);
                                } catch (error) {
                                    console.error("Error activating Feature:", error);
                                }
                            }
                        }}
                    >
                        {/* Equipment Card */}
                        <div className={`
              w-full rounded-xl overflow-hidden 
              border border-gray-800/30
              bg-black/90 backdrop-blur-lg
              ${hoveredItem === index ? getGlowColor(item.glow) : ''}
              transition-all duration-500 group
            `}>
                            {/* Icon Container */}
                            <div className={`
                h-48 relative flex items-center justify-center
                bg-gradient-to-br ${item.color}
                before:absolute before:inset-0 
                before:bg-[radial-gradient(circle_at_50%_120%,rgba(6,182,212,0.15),transparent)]
              `}>
                                <div className={`
                  transform transition-all duration-500
                  ${hoveredItem === index ? 'scale-110 rotate-6' : ''}
                  relative
                `}>
                                    {/* Enhanced Glow Effect */}
                                    <div className={`
                    absolute inset-0 blur-2xl opacity-40
                    ${getTextColor(item.glow).split(' ')[0].replace('text', 'bg')}
                  `} />
                                    <div className={`
                    relative z-10 transition-all duration-500
                    ${getTextColor(item.glow)}
                  `}>
                                        {item.icon}
                                    </div>
                                </div>

                                {/* Tech Lines Effect */}
                                <div className="absolute inset-0 overflow-hidden opacity-30">
                                    {[...Array(8)].map((_, i) => (
                                        <div
                                            key={i}
                                            className="absolute h-[1px] w-full bg-gradient-to-r from-transparent via-cyan-400 to-transparent transform -rotate-45"
                                            style={{
                                                top: `${i * 20}%`,
                                                left: '-100%',
                                                animation: `slideLines ${3 + i * 0.5}s linear infinite`
                                            }}
                                        />
                                    ))}
                                </div>
                            </div>

                            {/* Card Footer */}
                            <div className="p-6 bg-black/95 relative">
                                <h3 className={`
                  text-2xl font-bold font-mono mb-2 transition-colors duration-500
                  ${hoveredItem === index ? getTextColor(item.glow) : 'text-gray-300'}
                  drop-shadow-[0_0_8px_rgba(6,182,212,0.2)]
                `}>
                                    {item.name}
                                </h3>
                                <p className="text-gray-400 text-base font-mono mb-4 font-medium tracking-wide">
                                    {item.description}
                                </p>
                                <div className="flex gap-2">
                                    {item.indicators.map((color, iconIndex) => (
                                        <div
                                            key={iconIndex}
                                            className={`
                        w-6 h-6 rounded-full flex items-center justify-center bg-black/80
                        border border-gray-800/50
                        transition-all duration-500
                        ${hoveredItem === index ? 'scale-110 border-gray-700' : ''}
                      `}
                                        >
                                            <div className={`
                        w-2 h-2 rounded-full transition-transform duration-500
                        ${hoveredItem === index ? 'scale-125' : ''}
                        ${color === 'red' ? 'bg-red-500' : ''}
                        ${color === 'green' ? 'bg-green-500' : ''}
                        ${color === 'yellow' ? 'bg-yellow-500' : ''}
                        ${color === 'purple' ? 'bg-purple-500' : ''}
                        ${color === 'blue' ? 'bg-cyan-500' : ''}
                        ${color === 'cyan' ? 'bg-cyan-500' : ''}
                      `} />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <style jsx>{`
        @keyframes slideLines {
          0% { transform: translateX(0%) rotate(-45deg); }
          100% { transform: translateX(200%) rotate(-45deg); }
        }
      `}</style>
        </div>
    );
};

export default RetinaFeatures;