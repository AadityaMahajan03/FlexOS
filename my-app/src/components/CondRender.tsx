import React from 'react';

import BiometricDesktopUI from './BiometricDesktopUI'
import WindowsExplorer from './FileSystem/WindowsExplorer'
import HandGestures from './HandGestures'
import VoiceFeatures from './VoiceFeatures';
import RetinaFeatures from './RetinaFeatures';
import FacialFeatures from './FacialFeatures';
import ChatbotFeatures from './ChatbotFeatures';
import WindowsExplorerForFolders from './FileSystem/WindowsExplorerForFolders';
import WindowsExplorerForAllRest from './FileSystem/WindowsExplorerForAllRest';



const CondRender = ({ currentComponent, onNavigate }) => {
    const [component, root, filename] = currentComponent.split('_');
    currentComponent = component;
    switch (currentComponent) {
        case 'BiometricDesktopUI':
            return <BiometricDesktopUI onNavigate={onNavigate} />;
        case "WindowsExplorer":
            return <WindowsExplorer onNavigate={onNavigate} />;
        case "WindowsExplorerForFolders":
            return <WindowsExplorerForFolders onNavigate={onNavigate} filename={root} />;
        case "WindowsExplorerForAllRest":
            return <WindowsExplorerForAllRest onNavigate={onNavigate} root={root} filename={filename} />;
        case "HandGestures":
            return <HandGestures onNavigate={onNavigate} />;
        case "ChatbotFeatures":
            return <ChatbotFeatures onNavigate={onNavigate} />;
        case "VoiceFeatures":
            return <VoiceFeatures onNavigate={onNavigate} />;
        case "RetinaFeatures":
            return <RetinaFeatures onNavigate={onNavigate} />;
        case "FacialFeatures":
            return <FacialFeatures onNavigate={onNavigate} />;
        default:
            return <WindowsExplorer onNavigate={onNavigate} />;
    }
};

export default CondRender;