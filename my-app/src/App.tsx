import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import BiometricDesktopUI from './components/BiometricDesktopUI'
import WindowsExplorer from './components/WindowsExplorer'
import CondRender from './components/CondRender'

function App() {
  const [currentComponent, setCurrentComponent] = useState("BiometricDesktopUI");

  return (
    <>
      <div>
        <CondRender currentComponent={currentComponent} onNavigate={setCurrentComponent} />
      </div>
    </>
  )
}

export default App
