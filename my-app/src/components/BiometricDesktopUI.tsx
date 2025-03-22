import React, { useState, useEffect } from 'react';
import CondRender from './CondRender';
import {
    Monitor, Hand, Eye, Mic,
    User, MessageCircle, Power,
    Command, Shield, Sparkles,
    Activity, Lock, Check
} from 'lucide-react';

const BiometricDesktopUI = ({ onNavigate }) => {
    const [time, setTime] = useState(new Date().toLocaleTimeString());
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [activeFeature, setActiveFeature] = useState(null);
    const [activeDockItem, setActiveDockItem] = useState(null);
    const [isScanning, setIsScanning] = useState(false);

    useEffect(() => {
        const timer = setInterval(() => {
            setTime(new Date().toLocaleTimeString());
        }, 1000);

        const handleMouseMove = (e) => {
            setMousePos({ x: e.clientX, y: e.clientY });
        };

        window.addEventListener('mousemove', handleMouseMove);
        return () => {
            clearInterval(timer);
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);


    const features = [
        {
            name: 'This PC',
            icon: <Monitor className="w-8 h-8" />,
            color: 'from-blue-500 to-cyan-500',
            description: 'Access System Core',
            status: 'Connected',
            componentName: 'WindowsExplorer',

        },
        {
            name: 'Gestura',
            icon: <Hand className="w-8 h-8" />,
            color: 'from-purple-500 to-pink-500',
            description: 'Hand Gesture Control',
            status: 'Learning',
            componentName: 'HandGestures',
        },
        {
            name: 'IrisKey',
            icon: <Eye className="w-8 h-8" />,
            color: 'from-emerald-500 to-teal-500',
            description: 'Retina Based Security',
            status: 'Scanning',
            componentName: 'RetinaFeatures',
        },
        {
            name: 'EchoVault',
            icon: <Mic className="w-8 h-8" />,
            color: 'from-red-500 to-orange-500',
            description: 'Voice Authentication',
            status: 'Listening',
            componentName: 'VoiceFeatures',
        },
        {
            name: 'FaceVault',
            icon: <User className="w-8 h-8" />,
            color: 'from-yellow-500 to-amber-500',
            description: 'Facial Recognition',
            status: 'Active',
            componentName: 'FacialFeatures',
        },
        {
            name: 'PolyTalk',
            icon: <MessageCircle className="w-8 h-8" />,
            color: 'from-indigo-500 to-purple-500',
            description: 'Multilingual AI Assistant',
            status: 'Ready',
            componentName: 'ChatbotFeatures',
        }
    ];

    return (
        <div className="min-h-screen bg-gray-900 text-white overflow-hidden relative">
            {/* Enhanced Dynamic Background */}
            <div className="absolute inset-0">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,_rgba(17,24,39,1),rgba(17,24,39,0.8))]" />
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5" />
            </div>

            {/* Enhanced Interactive Glow Effect */}
            <div
                className="absolute blur-3xl opacity-30 transition-all duration-700 ease-out"
                style={{
                    background: 'radial-gradient(circle at center, #4F46E5, transparent 70%)',
                    width: '40rem',
                    height: '40rem',
                    transform: `translate(${mousePos.x - 320}px, ${mousePos.y - 320}px)`,
                    pointerEvents: 'none'
                }}
            />

            {/* Improved Top Bar */}
            <div className="fixed top-0 w-full h-16 backdrop-blur-2xl bg-black/20 border-b border-white/10 z-50">
                <div className="flex justify-between items-center h-full px-6">
                    <div className="flex items-center gap-4">
                        <div className="relative">
                            <Command className="w-6 h-6 text-blue-400" />
                            <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full animate-ping" />
                        </div>
                        <span className="text-lg font-semibold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
                            FlexOS: Universal Operating System Simulation
                        </span>
                    </div>
                    <div className="flex items-center gap-6">
                        <div className="flex gap-3">
                            {Array(3).fill(0).map((_, i) => (
                                <div key={i}
                                    className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"
                                    style={{
                                        animationDelay: `${i * 200}ms`,
                                        animationDuration: '1.5s'
                                    }}
                                />
                            ))}
                        </div>
                        <div className="font-mono text-lg bg-white/5 px-4 py-1.5 rounded-lg border border-white/10 relative group">
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg" />
                            {time}
                        </div>
                    </div>
                </div>
            </div>

            {/* Enhanced Main Grid */}
            <div className="pt-24 p-8 grid grid-cols-3 gap-8">
                {features.map((feature, index) => (
                    <div key={index}
                        className="group"
                        onMouseEnter={() => setActiveFeature(feature.name)}
                        onMouseLeave={() => setActiveFeature(null)}
                        onClick={() => onNavigate(feature.componentName)}
                    >
                        <div className={`relative rounded-2xl backdrop-blur-xl bg-black/30 border border-white/10 
                            p-8 transition-all duration-500 hover:scale-105 hover:bg-black/40 
                            hover:border-white/20 cursor-pointer overflow-hidden`}>
                            {/* Enhanced Animated Gradient Background */}
                            <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 
                                group-hover:opacity-20 transition-opacity duration-500`} />

                            {/* Enhanced Glow Effect */}
                            <div className={`absolute -inset-px bg-gradient-to-r ${feature.color} opacity-0 
                                group-hover:opacity-30 blur-xl transition-opacity duration-300`} />

                            {/* Improved Content */}
                            <div className="relative z-10">
                                <div className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${feature.color} 
                                    flex items-center justify-center mb-6 shadow-lg group-hover:shadow-xl 
                                    transition-all duration-300 relative overflow-hidden`}>
                                    {feature.icon}
                                    <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 
                                        transition-opacity duration-300" />
                                </div>
                                <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                                    {feature.name}
                                    <span className="text-sm font-normal text-white/60">
                                        {feature.status}
                                    </span>
                                </h3>
                                <div className="flex items-center gap-2 text-white/60">
                                    <Sparkles className="w-4 h-4" />
                                    <span>{feature.description}</span>
                                </div>

                                {/* Enhanced Status Indicator */}
                                <div className="absolute top-4 right-4 flex items-center gap-2">
                                    <span className="text-xs text-white/40 opacity-0 group-hover:opacity-100 
                                        transition-opacity duration-300">
                                        {feature.status}
                                    </span>
                                    <div className="w-3 h-3 rounded-full bg-gradient-to-r from-green-400 to-emerald-500 
                                        animate-pulse shadow-lg shadow-green-500/50 group-hover:scale-110 
                                        transition-transform duration-300" />
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Enhanced Dock */}
            <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2">
                <div className="flex gap-4 p-4 rounded-2xl backdrop-blur-3xl bg-black/40 
                    border border-white/10 shadow-2xl shadow-blue-500/20">
                    {features.map((feature, index) => (
                        <button
                            key={index}
                            className="group relative"
                            onMouseEnter={() => setActiveDockItem(feature.name)}
                            onMouseLeave={() => setActiveDockItem(null)}
                        >
                            <div className={`absolute inset-0 bg-gradient-to-r ${feature.color} opacity-0 
                                group-hover:opacity-100 blur-xl transition-opacity duration-300 -z-10`} />

                            <div className={`p-3 rounded-xl bg-gradient-to-br ${feature.color} 
                                transform transition-all duration-300 group-hover:scale-110 
                                group-hover:shadow-lg relative`}>
                                {feature.icon}
                                {activeDockItem === feature.name && (
                                    <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 
                                        bg-black/80 text-white px-2 py-1 rounded text-sm whitespace-nowrap">
                                        {feature.name}
                                    </div>
                                )}
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Enhanced System Status */}
            <div className="fixed right-8 top-24 w-64 backdrop-blur-3xl bg-black/30 
                border border-white/10 rounded-2xl shadow-2xl shadow-blue-500/20">
                <div className="p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <Shield className="w-5 h-5 text-blue-400" />
                        System Status
                    </h3>
                    <div className="space-y-4">
                        {activeFeature && (
                            <div className="animate-fadeIn">
                                <div className="text-sm text-white/60 mb-2 flex justify-between">
                                    <span>{activeFeature} Active</span>
                                    <span className="text-green-400">
                                        <Check className="w-4 h-4" />
                                    </span>
                                </div>
                                <div className="w-full h-1.5 bg-white/20 rounded-full overflow-hidden">
                                    <div className="h-full w-4/5 bg-blue-500 rounded-full animate-pulse" />
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BiometricDesktopUI;