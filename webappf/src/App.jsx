import { useState, useEffect } from 'react';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  // --- API BRIDGE ---
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
    fetchData();
    const interval = setInterval(fetchData, 1000); 
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div className="min-h-screen bg-black text-green-500 flex items-center justify-center font-mono text-xl">CONNECTING TO SATELLITE...</div>;

  const isEmergency = data.emergency_detected;

  return (
    <div className="min-h-screen bg-slate-950 text-white font-sans selection:bg-green-500 selection:text-black">
      
      {/* --- COMMAND BAR --- */}
      <nav className="h-16 border-b border-slate-800 bg-black/80 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse shadow-[0_0_10px_red]"></div>
          <h1 className="text-xl font-bold tracking-widest text-slate-200">
            TRAFFIC<span className="text-blue-500">PULSE</span> <span className="text-xs text-slate-600 border border-slate-700 px-2 py-0.5 rounded">V8.2.0</span>
          </h1>
        </div>
        <div className="font-mono text-xs text-slate-500 flex gap-6">
          <span>CPU: <span className="text-green-400">NORMAL</span></span>
          <span>LATENCY: <span className="text-green-400">12ms</span></span>
        </div>
      </nav>

      <main className="max-w-[1600px] mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* --- LEFT COLUMN: LIVE VISION (Spans 8 cols) --- */}
        <div className="lg:col-span-8 flex flex-col gap-4">
          
          {/* VIDEO PLAYER CONTAINER */}
          <div className="relative aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 shadow-2xl group">
            
            {/* The Live Stream Image */}
            <img 
              src="http://localhost:5000/video_feed" 
              alt="Live AI Feed" 
              className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
            />

            {/* AI Overlay Graphics */}
            <div className="absolute top-4 left-4 flex gap-2">
              <span className="bg-red-600 text-white text-[10px] font-bold px-2 py-1 rounded">LIVE FEED</span>
              <span className="bg-black/60 backdrop-blur text-white text-[10px] font-mono px-2 py-1 rounded border border-white/10">CAM-01</span>
            </div>
            
            <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end">
               <div className="text-[10px] font-mono text-green-400 bg-black/80 px-2 py-1 rounded">
                 OBJECTS TRACKED: {data.waiting_cars.north + data.waiting_cars.south + data.waiting_cars.east + data.waiting_cars.west}
               </div>
               <div className="text-[10px] font-mono text-blue-400 bg-black/80 px-2 py-1 rounded animate-pulse">
                 SCANNING...
               </div>
            </div>

            {/* Scanline Effect */}
            <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] pointer-events-none opacity-20"></div>
          </div>

          {/* SYSTEM STATUS BANNER */}
          <div className={`p-4 rounded-lg border flex items-center justify-between transition-all duration-500
            ${isEmergency ? 'bg-red-900/20 border-red-500/50' : 'bg-green-900/10 border-green-500/30'}`}>
            <div className="flex items-center gap-4">
               <div className={`text-3xl ${isEmergency ? 'animate-bounce' : ''}`}>{isEmergency ? '🚨' : '🛡️'}</div>
               <div>
                 <h2 className={`font-bold ${isEmergency ? 'text-red-400' : 'text-green-400'}`}>
                   {isEmergency ? 'CRITICAL ALERT: EMERGENCY VEHICLE' : 'SYSTEM OPTIMIZED'}
                 </h2>
                 <p className="text-xs text-slate-400">
                   {isEmergency ? 'Rerouting traffic. Clearing North/South corridor.' : 'Traffic flow operating within normal parameters.'}
                 </p>
               </div>
            </div>
            <button className="bg-slate-800 hover:bg-slate-700 text-xs px-4 py-2 rounded border border-slate-600 transition-colors">
              VIEW LOGS
            </button>
          </div>
        </div>


        {/* --- RIGHT COLUMN: ANALYTICS (Spans 4 cols) --- */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
            <h3 className="text-slate-500 text-xs font-bold uppercase mb-4 tracking-wider">Traffic Density</h3>
            <div className="space-y-4">
              <QueueBar label="North" value={data.waiting_cars.north} color="bg-blue-500" />
              <QueueBar label="South" value={data.waiting_cars.south} color="bg-blue-500" />
              <QueueBar label="East"  value={data.waiting_cars.east}  color="bg-purple-500" />
              <QueueBar label="West"  value={data.waiting_cars.west}  color="bg-purple-500" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
             <StatCard label="Phase" value={data.current_green_phase} />
             <StatCard label="Wait Time" value={`${data.avg_wait_time}s`} />
          </div>

          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex-grow flex flex-col justify-center items-center text-center opacity-60">
             <div className="text-4xl mb-2">☁️</div>
             <div className="text-sm font-bold text-slate-400">Weather API</div>
             <div className="text-xs text-slate-600">Offline</div>
          </div>

        </div>

      </main>
    </div>
  );
}

// Sub-components for cleaner code
function QueueBar({ label, value, color }) {
  const width = Math.min(value * 4, 100); // Scale bar width
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label} Queue</span>
        <span className="font-mono">{value} cars</span>
      </div>
      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all duration-1000`} style={{ width: `${width}%` }}></div>
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
      <div className="text-slate-500 text-[10px] font-bold uppercase">{label}</div>
      <div className="text-lg font-mono text-white mt-1">{value}</div>
    </div>
  )
}

export default App;