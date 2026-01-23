import { useState, useEffect } from 'react';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState([]);
  const [isBooting, setIsBooting] = useState(true);

  // --- 1. INITIAL SYSTEM BOOT SEQUENCE ---
  useEffect(() => {
    setTimeout(() => setIsBooting(false), 2000);
  }, []);

  // --- 2. MAIN DATA POLLING ---
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/status');
        if (res.ok) {
          const jsonData = await res.json();
          setData(jsonData);
          setLoading(false);
        }
      } catch (err) {
        console.error("API Error", err);
      }
    };
    if (!isBooting) {
      fetchData();
      const interval = setInterval(fetchData, 500);
      return () => clearInterval(interval);
    }
  }, [isBooting]);

  // --- 3. FETCH LOGS FUNCTION ---
  const fetchLogs = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/logs');
      const jsonLogs = await res.json();
      setLogs(jsonLogs);
    } catch (err) {
      console.error("Log Fetch Error", err);
      setLogs([]); 
    }
  };

  const handleOpenLogs = () => {
    setShowLogs(true);
    fetchLogs();
  };

  // --- RENDER LOADING / BOOT SCREEN ---
  if (loading || isBooting) return (
    <div className="min-h-screen bg-black flex flex-col items-center justify-center font-mono text-green-500">
      <div className="animate-spin text-4xl mb-4">⚙️</div>
      <div className="text-xl tracking-[0.5em] animate-pulse">INITIALIZING DUAL-SENSOR SYSTEM...</div>
      <div className="text-xs text-green-800 mt-2">CALIBRATING AUDIO & VISION SENSORS</div>
    </div>
  );

  const isEmergency = data.emergency_detected;
  const isAudioAlert = data.audio_alert;
  
  // LOGIC: Determine which light is green. 
  // If Emergency is active, ALL might turn red or specific logic applies.
  // For now, we follow the user rule: based on current_green_phase.
  const activePhase = isEmergency ? 'EMERGENCY_OVERRIDE' : (data.current_green_phase || 'North'); 

  return (
    <div className="min-h-screen bg-slate-950 text-white font-sans selection:bg-blue-500 selection:text-white overflow-hidden relative">
      
      {/* Background Grid Effect */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(18,18,27,1)_2px,transparent_2px),linear-gradient(90deg,rgba(18,18,27,1)_2px,transparent_2px)] bg-[length:40px_40px] opacity-20 pointer-events-none"></div>

      {/* --- TOP COMMAND BAR --- */}
      <nav className="h-16 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-40 shadow-lg">
        <div className="flex items-center gap-4">
          <div className={`w-3 h-3 rounded-full ${isEmergency ? 'bg-red-500 shadow-[0_0_15px_red] animate-ping' : 'bg-green-500 shadow-[0_0_10px_green]'}`}></div>
          <h1 className="text-2xl font-black tracking-widest text-white">
            TRAFFIC<span className="text-blue-500">PULSE</span>
            <span className="ml-3 text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">DUAL-SENSOR V2.0</span>
          </h1>
        </div>
        <div className="hidden md:flex font-mono text-xs text-slate-500 gap-8">
           <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isAudioAlert ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></span>
              <span>AUDIO: <span className={isAudioAlert ? 'text-red-400 font-bold' : 'text-slate-400'}>{isAudioAlert ? 'DETECTED' : 'LISTENING'}</span></span>
           </div>
           <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              <span>VISION: ACTIVE</span>
           </div>
        </div>
      </nav>

      <main className="max-w-[1800px] mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
        
        {/* --- LEFT COLUMN: VISION & CONTROLS (8 Cols) --- */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          
          {/* VIDEO FEED */}
          <div className="relative aspect-video bg-black rounded-2xl overflow-hidden border border-slate-800 shadow-[0_0_50px_rgba(0,0,0,0.5)] group">
            <img 
              src="http://localhost:5000/video_feed" 
              alt="Live AI Feed" 
              className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity duration-500"
            />
            
            <div className="absolute inset-0 pointer-events-none border-[1px] border-white/5 rounded-2xl">
              {isAudioAlert && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center justify-center animate-pulse z-50">
                   <div className="text-6xl mb-2">🔊</div>
                   <div className="bg-red-600 text-white font-black text-xl px-4 py-2 uppercase tracking-widest border-2 border-white shadow-[0_0_20px_red]">
                     Acoustic Siren Match
                   </div>
                </div>
              )}
              <div className="absolute top-4 left-4 flex gap-2">
                <span className="bg-red-600 text-white text-[10px] font-bold px-3 py-1 rounded-sm shadow-lg animate-pulse">LIVE</span>
                <span className="bg-black/60 backdrop-blur-md text-slate-300 text-[10px] font-mono px-3 py-1 rounded-sm border border-white/10">CAM_FEED_01</span>
              </div>
            </div>
          </div>

          {/* STATUS CONTROL PANEL */}
          <div className={`relative p-6 rounded-2xl border transition-all duration-500 overflow-hidden flex flex-col gap-4
            ${isEmergency 
              ? 'bg-gradient-to-r from-red-950/50 to-slate-950 border-red-500/50 shadow-[0_0_30px_rgba(220,38,38,0.2)]' 
              : 'bg-slate-900/50 border-slate-800'
            }`}>
            
            {isEmergency && <div className="absolute inset-0 bg-red-500/10 animate-pulse pointer-events-none"></div>}

            <div className="relative flex items-center justify-between z-10">
              <div className="flex items-center gap-6">
                <div className={`w-16 h-16 rounded-xl flex items-center justify-center text-3xl shadow-inner
                  ${isEmergency ? 'bg-red-500/20 text-red-500' : 'bg-green-500/10 text-green-500'}`}>
                  {isAudioAlert ? '🔊' : (isEmergency ? '🚨' : '🛡️')}
                </div>
                <div>
                  <h2 className={`text-2xl font-bold tracking-tight mb-1 ${isEmergency ? 'text-red-400' : 'text-green-400'}`}>
                    {isAudioAlert ? 'AUDIO SIREN CONFIRMED' : (isEmergency ? 'EMERGENCY DETECTED' : 'SYSTEM OPTIMIZED')}
                  </h2>
                  <p className="text-sm text-slate-400 font-mono">
                    {isEmergency 
                      ? 'Priority Override Active. Clearing Path.' 
                      : 'Monitoring Audio & Visual Channels.'}
                  </p>
                </div>
              </div>

              <button 
                onClick={handleOpenLogs}
                className="group relative px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold tracking-wider uppercase rounded-lg border border-slate-700 transition-all hover:border-slate-500 hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0"
              >
                <span className="flex items-center gap-2">
                  <span>View Logs</span>
                  <span className="group-hover:translate-x-1 transition-transform">→</span>
                </span>
              </button>
            </div>

            <div className="h-12 bg-black/40 rounded-lg border border-slate-800 flex items-center justify-center gap-1 px-4 overflow-hidden relative">
                {[...Array(50)].map((_, i) => (
                    <div 
                        key={i} 
                        className={`w-1 rounded-full transition-all duration-300 ${isAudioAlert ? 'bg-red-500 animate-pulse' : 'bg-green-900'}`}
                        style={{ 
                            height: isAudioAlert ? `${((i * 13) % 60) + 20}%` : '20%',
                            opacity: isAudioAlert ? 1 : 0.3
                        }}
                    ></div>
                ))}
            </div>
            <div className="text-[10px] text-center font-mono text-slate-500 -mt-2 uppercase tracking-widest">Microphone Input Spectrum</div>

          </div>
        </div>

        {/* --- RIGHT COLUMN: ANALYTICS (4 Cols) --- */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          
          {/* --- NEW: INTERSECTION VISUALIZATION --- */}
          <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
             <div className="flex justify-between items-center mb-4">
                <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest">Live Junction Logic</h3>
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
             </div>

             {/* THE 4-WAY INTERSECTION GRID */}
             <div className="flex flex-col items-center justify-center py-4 relative h-64">
                 
                 {/* CSS ROADS (The "4 Lines Intersecting") */}
                 <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="w-16 h-full bg-slate-800/50 border-x border-slate-700/30"></div> {/* Vertical Road */}
                    <div className="h-16 w-full bg-slate-800/50 border-y border-slate-700/30"></div> {/* Horizontal Road */}
                 </div>

                 {/* Top (North) */}
                 <div className="absolute top-0">
                    <TrafficLight 
                        label="NORTH" 
                        isActive={activePhase.toLowerCase() === 'north'} 
                        cars={data.waiting_cars.north} 
                    />
                 </div>

                 {/* Left (West) */}
                 <div className="absolute left-0 top-1/2 -translate-y-1/2">
                    <TrafficLight 
                        label="WEST" 
                        isActive={activePhase.toLowerCase() === 'west'} 
                        cars={data.waiting_cars.west} 
                    />
                 </div>

                 {/* Right (East) */}
                 <div className="absolute right-0 top-1/2 -translate-y-1/2">
                    <TrafficLight 
                        label="EAST" 
                        isActive={activePhase.toLowerCase() === 'east'} 
                        cars={data.waiting_cars.east} 
                    />
                 </div>

                 {/* Bottom (South) */}
                 <div className="absolute bottom-0">
                    <TrafficLight 
                        label="SOUTH" 
                        isActive={activePhase.toLowerCase() === 'south'} 
                        cars={data.waiting_cars.south} 
                    />
                 </div>
                 
                 {/* Center Label */}
                 <div className="z-10 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[10px] font-mono text-slate-500">
                    INT-01
                 </div>

             </div>
             <div className="text-[10px] text-center text-slate-500 mt-4 font-mono">
                LOGIC: <span className="text-white">{activePhase.toUpperCase()} IS GREEN</span>
             </div>
          </div>


          {/* TRAFFIC DENSITY CARD */}
          <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-6">Queue Density</h3>
            <div className="space-y-6">
              <QueueBar label="North Lane" value={data.waiting_cars.north} color="bg-blue-500" />
              <QueueBar label="South Lane" value={data.waiting_cars.south} color="bg-blue-500" />
              <QueueBar label="East Lane"  value={data.waiting_cars.east}  color="bg-purple-500" />
              <QueueBar label="West Lane"  value={data.waiting_cars.west}  color="bg-purple-500" />
            </div>
          </div>

          <div className="bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800 rounded-2xl p-6 flex flex-col items-center justify-center text-center relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-3 opacity-50 group-hover:opacity-100 transition-opacity">
               <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
            </div>
            <div className="text-5xl mb-3 grayscale group-hover:grayscale-0 transition-all duration-700">⛅</div>
            <div className="text-sm font-bold text-slate-300">Local Weather</div>
            <div className="text-xs text-slate-500 mt-1">24°C • Partly Cloudy • Wind 12km/h</div>
          </div>

        </div>
      </main>

      {/* --- LOGS MODAL --- */}
      {showLogs && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200">
          <div className="bg-slate-900 border border-slate-700 w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[80vh]">
            <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950">
              <h3 className="font-mono text-white font-bold flex items-center gap-2"><span>📂</span> SYSTEM EVENT LOGS</h3>
              <button onClick={() => setShowLogs(false)} className="text-slate-500 hover:text-white transition-colors">✕ CLOSE</button>
            </div>
            <div className="p-0 overflow-y-auto font-mono text-xs flex-grow bg-slate-950/50">
              <table className="w-full text-left">
                <thead className="bg-slate-900 text-slate-400 sticky top-0">
                  <tr><th className="p-3">TIMESTAMP</th><th className="p-3">EVENT TYPE</th><th className="p-3">CONFIDENCE</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-300">
                  {logs.length === 0 ? (
                    <tr><td colSpan="4" className="p-8 text-center text-slate-600">No events recorded yet...</td></tr>
                  ) : (
                    logs.map((log, i) => (
                      <tr key={i} className="hover:bg-slate-800/50 transition-colors">
                        <td className="p-3">{log[1]}</td>
                        <td className={`p-3 ${log[2].includes('AUDIO') ? 'text-pink-400 font-bold' : 'text-yellow-400'}`}>{log[2]}</td>
                        <td className="p-3 text-green-400">{(log[3] * 100).toFixed(1)}%</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
            <div className="p-3 border-t border-slate-800 bg-slate-900 flex justify-end">
               <button onClick={fetchLogs} className="text-xs text-blue-400 hover:text-blue-300">REFRESH DATA ↻</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// --- SUB-COMPONENTS ---

function TrafficLight({ label, isActive, cars }) {
    return (
        <div className={`flex flex-col items-center p-2 rounded-lg border transition-all duration-500 w-20 bg-slate-900
            ${isActive 
                ? 'border-green-500 shadow-[0_0_15px_rgba(34,197,94,0.3)] scale-110 z-10' 
                : 'border-slate-700 opacity-70 scale-90'
            }`}>
            <div className="text-[9px] font-bold text-slate-400 mb-1">{label}</div>
            
            {/* Light Bulb */}
            <div className={`w-6 h-6 rounded-full mb-1 shadow-inner border border-black/50 transition-colors duration-300
                ${isActive 
                    ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' 
                    : 'bg-red-600 shadow-[0_0_5px_#dc2626]'
                }`}>
            </div>

            <div className="font-mono text-[9px] text-white">{cars} Cars</div>
        </div>
    );
}

function QueueBar({ label, value, color }) {
  const width = Math.min(value * 4, 100); 
  return (
    <div className="group">
      <div className="flex justify-between text-[10px] uppercase font-bold text-slate-500 mb-2 group-hover:text-slate-300 transition-colors">
        <span>{label}</span><span className="font-mono text-white">{value} CARS</span>
      </div>
      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
        <div className={`h-full ${color} shadow-[0_0_10px_currentColor] transition-all duration-1000 ease-out`} style={{ width: `${width}%` }}></div>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon, subtext }) {
  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 hover:bg-slate-800/50 transition-colors group">
      <div className="flex justify-between items-start mb-2">
        <div className="text-slate-500 text-[10px] font-bold uppercase tracking-wider">{label}</div>
        <div className="text-xl opacity-50 group-hover:opacity-100 transition-opacity group-hover:scale-110 duration-300">{icon}</div>
      </div>
      <div className="text-2xl font-mono font-bold text-white mb-1">{value}</div>
      <div className="text-[10px] text-slate-600 group-hover:text-slate-400">{subtext}</div>
    </div>
  )
}

export default App;
