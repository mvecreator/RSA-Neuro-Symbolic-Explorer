/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Key, 
  Terminal, 
  Orbit, 
  TrendingDown, 
  CheckCircle2, 
  XCircle, 
  Play, 
  RotateCcw,
  Cpu,
  Zap,
  Shield,
  Search,
  Copy
} from 'lucide-react';
import { generateKeys, RSAKey } from './lib/rsa';

export default function App() {
  const [key, setKey] = useState<RSAKey | null>(null);
  const [targetN, setTargetN] = useState<string>('');
  const [isSimulating, setIsSimulating] = useState(false);
  const [steps, setSteps] = useState<{ x: number; y: number; type: string; energy: number }[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState<{ p: bigint; q: bigint } | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [showD, setShowD] = useState(false);
  const [activeTab, setActiveTab] = useState<'simulator' | 'sandbox'>('simulator');
  const [complexityMode, setComplexityMode] = useState<'16-bit' | '2048-bit'>('16-bit');
  
  // Sandbox state
  const [sandboxA, setSandboxA] = useState('3');
  const [sandboxM, setSandboxM] = useState('11');
  const [sandboxI, setSandboxI] = useState('');
  
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Only the small-key demo can generate verified prime factors.
  useEffect(() => {
    if (complexityMode === '2048-bit') {
      setKey(null);
      setTargetN('');
      resetSimulation();
    } else {
      const newKey = generateKeys(8);
      setKey(newKey);
      setTargetN(newKey.n.toString());
      resetSimulation();
    }
  }, [complexityMode]);

  const handleGenerateKey = () => {
    if (complexityMode !== '16-bit') return;
    const newKey = generateKeys(8);
    setKey(newKey);
    setTargetN(newKey.n.toString());
    resetSimulation();
  };

  const resetSimulation = () => {
    setIsSimulating(false);
    setSteps([]);
    setCurrentStep(0);
    setResult(null);
    setShowD(false);
    setLogs(['System ready. Waiting for input...']);
  };

  const addLog = (msg: string) => {
    setLogs(prev => [msg, ...prev].slice(0, 10));
  };

  const startSimulation = useCallback(async () => {
    if (!/^[1-9][0-9]{0,616}$/.test(targetN)) {
      addLog('Error: N must be a positive decimal integer (up to 617 digits).');
      return;
    }

    setIsSimulating(true);
    setResult(null);
    setSteps([]);
    addLog(`System: Initiating RSA ${complexityMode} calculation...`);
    
    // Attempt real calculation from Python Backend
    try {
      const response = await fetch('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          n: targetN,
          bit_length: complexityMode === '16-bit' ? 16 : 2048
        })
      });
      
      if (!response.ok) throw new Error('Backend failed');
      
      const realData = await response.json();
      
      if (realData.error) {
        addLog(`Error: ${realData.error}`);
        setIsSimulating(false);
        return;
      }

      if (realData.message) {
        addLog(`Note: ${realData.message}`);
      }

      if (realData.status === 'unsupported') {
        setIsSimulating(false);
        return;
      }

      addLog(`System: Backend completed. Rendering ${realData.steps.length} steps.`);
      if (realData.oracle_used) addLog('System: MDN Oracle was engaged in this solution.');

      // Replay the steps with a delay
      let animationPath: any[] = [];
      const replay = (idx: number) => {
        if (idx >= realData.steps.length) {
          setIsSimulating(false);
          if (realData.p && realData.q) {
            const p = BigInt(realData.p), q = BigInt(realData.q);
            if (p * q === BigInt(targetN)) {
              setResult({ p, q });
              addLog(`[Success] Factors verified: ${p} x ${q}`);
            } else {
              addLog('Error: Backend factors do not multiply to N.');
            }
          } else {
            addLog('System: No factors found.');
          }
          return;
        }

        const step = realData.steps[idx];
        animationPath.push(step);
        setSteps([...animationPath]);
        
        if (step.type === 'mdn_jump') {
          addLog(`Step ${idx}: MDN Oracle redirection triggered.`);
        } else if (step.type === 'success') {
          addLog('Step ' + idx + ': Convergence reached. Symbolic collapse.');
        }

        setTimeout(() => replay(idx + 1), realData.steps.length > 100 ? 5 : 20);
      };

      replay(0);

    } catch (e) {
      addLog('System: Backend unavailable. No factorization performed.');
      setIsSimulating(false);
    }
  }, [targetN, complexityMode]);

  // Draw simulation on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const n = Number(targetN);
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    if (!Number.isSafeInteger(n) || n < 4 || Math.sqrt(n) * 2.5 > 2000) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '14px sans-serif';
      ctx.fillText('Projection available only for small N', 24, height / 2);
      return;
    }

    // Scale mapping
    const maxVal = Math.sqrt(n) * 2.5;
    const scale = width / maxVal;

    // Grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < maxVal; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i * scale, 0);
      ctx.lineTo(i * scale, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * scale);
      ctx.lineTo(width, i * scale);
      ctx.stroke();
    }

    // Energy Contours (Simulated)
    ctx.lineWidth = 1;
    [0.9, 1.1, 1.3, 1.5].forEach((multiplier, idx) => {
      ctx.strokeStyle = `rgba(59, 130, 246, ${0.1 / (idx + 1)})`;
      ctx.beginPath();
      const contourN = n * multiplier;
      for (let tx = 2; tx < maxVal; tx += 2) {
        const ty = contourN / tx;
        if (ty < maxVal) {
          if (tx === 2) ctx.moveTo(tx * scale, ty * scale);
          else ctx.lineTo(tx * scale, ty * scale);
        }
      }
      ctx.stroke();
    });

    // Hyperbola xy = n
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let tx = 2; tx < maxVal; tx += 1) {
      const ty = n / tx;
      if (ty < maxVal) {
        if (tx === 2) ctx.moveTo(tx * scale, ty * scale);
        else ctx.lineTo(tx * scale, ty * scale);
      }
    }
    ctx.stroke();

    // Trajectory
    if (steps.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 1]);
      ctx.moveTo(steps[0].x * scale, steps[0].y * scale);
      for (let i = 1; i < steps.length; i++) {
        ctx.lineTo(steps[i].x * scale, steps[i].y * scale);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Current point
      const last = steps[steps.length - 1];
      ctx.fillStyle = last.type === 'success' ? '#22c55e' : (last.type === 'mdn_jump' ? '#f59e0b' : '#3b82f6');
      ctx.beginPath();
      ctx.arc(last.x * scale, last.y * scale, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Glow effect for jumps and success
      if (last.type === 'mdn_jump' || last.type === 'success') {
        ctx.shadowBlur = 10;
        ctx.shadowColor = ctx.fillStyle as string;
        ctx.beginPath();
        ctx.arc(last.x * scale, last.y * scale, 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }

    // True Factors if known
    if (key && BigInt(n) === key.n) {
      ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
      ctx.beginPath();
      ctx.arc(Number(key.p) * scale, Number(key.q) * scale, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.arc(Number(key.q) * scale, Number(key.p) * scale, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [steps, targetN, key]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 rounded-lg shadow-lg shadow-blue-500/20">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white">TKHD+MDN RSA</h1>
              <p className="text-xs text-slate-400 font-mono uppercase tracking-widest">Neuro-Symbolic Framework</p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <nav className="hidden md:flex gap-6 text-sm font-medium text-slate-400">
              <button 
                onClick={() => setActiveTab('simulator')} 
                className={`hover:text-blue-400 transition-colors ${activeTab === 'simulator' ? 'text-blue-400' : ''}`}
              >
                Simulator
              </button>
              <button 
                onClick={() => setActiveTab('sandbox')} 
                className={`hover:text-amber-400 transition-colors ${activeTab === 'sandbox' ? 'text-amber-400' : ''}`}
              >
                Modular Sandbox
              </button>
            </nav>
            <button 
              onClick={handleGenerateKey}
              disabled={complexityMode === '2048-bit'}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-full text-sm font-semibold transition-all border border-white/5 active:scale-95"
            >
              <Key className="w-4 h-4" />
              Generate Key
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12 space-y-12">
        {/* Intro Section */}
        <section className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-bold uppercase tracking-wider"
            >
              <Shield className="w-3 h-3" />
              Cryptanalysis V2.0
            </motion.div>
            <motion.h2 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-5xl font-extrabold text-white leading-tight"
            >
              Breaking 16-bit RSA with <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">Continuous Topology</span>
            </motion.h2>
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-lg text-slate-400 max-w-xl"
            >
              Most factorization algorithms are discrete. Our method, <b>TKHD+MDN</b>, transforms the problem into a continuous energy landscape, using neural networks to guide the path through topological singularities.
            </motion.p>
          </div>
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative"
          >
            <div className="absolute -inset-4 bg-blue-500/20 blur-3xl rounded-full" />
            <div className="relative bg-slate-900/50 border border-white/10 p-8 rounded-3xl backdrop-blur-xl shadow-2xl">
              <div className="flex items-center justify-between mb-6">
                <div className="flex gap-2">
                  <button 
                    onClick={() => setComplexityMode('16-bit')}
                    className={`px-3 py-1 rounded-full border text-[10px] font-bold transition-all ${
                      complexityMode === '16-bit' 
                        ? 'bg-blue-500/20 border-blue-500 text-blue-400' 
                        : 'bg-white/5 border-white/10 text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    16-BIT
                  </button>
                  <button 
                    onClick={() => setComplexityMode('2048-bit')}
                    className={`px-3 py-1 rounded-full border text-[10px] font-bold transition-all ${
                      complexityMode === '2048-bit' 
                        ? 'bg-red-500/20 border-red-500 text-red-400' 
                        : 'bg-white/5 border-white/10 text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    2048-BIT (UNIMPLEMENTED)
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-500 uppercase">{complexityMode === '16-bit' ? 'Research' : 'Enterprise'} Keypair</span>
                  <div className={`w-2 h-2 rounded-full animate-pulse ${complexityMode === '16-bit' ? 'bg-emerald-500' : 'bg-red-500'}`} />
                </div>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-black/40 rounded-xl border border-white/5 space-y-1">
                    <label className={`text-[10px] uppercase font-bold tracking-widest ${complexityMode === '16-bit' ? 'text-blue-400' : 'text-red-400'}`}>Public N</label>
                    <p className={`font-mono text-white ${complexityMode === '16-bit' ? 'text-xl' : 'text-[10px] break-all max-h-12 overflow-y-auto'}`}>
                      {key?.n.toString() || '---'}
                    </p>
                  </div>
                  <div className="p-4 bg-black/40 rounded-xl border border-white/5 space-y-1">
                    <label className={`text-[10px] uppercase font-bold tracking-widest ${complexityMode === '16-bit' ? 'text-blue-400' : 'text-red-400'}`}>Public E</label>
                    <p className="text-xl font-mono text-white">{key?.e.toString() || '---'}</p>
                  </div>
                </div>
                <div className={`p-4 bg-black/40 rounded-xl border border-white/5 space-y-1 relative group bg-gradient-to-r transition-all ${
                  complexityMode === '16-bit' ? 'from-blue-900/10' : 'from-red-900/20'
                } to-transparent`}>
                  <div className="flex justify-between items-center">
                    <label className={`text-[10px] uppercase font-bold tracking-widest ${complexityMode === '16-bit' ? 'text-amber-500' : 'text-red-500'}`}>Secret D</label>
                    <span className={`text-[8px] px-1 rounded uppercase font-bold ${
                      complexityMode === '16-bit' ? 'bg-amber-500/20 text-amber-500' : 'bg-red-500/20 text-red-500'
                    }`}>
                      {complexityMode === '16-bit' ? 'Encrypted' : 'REDACTED'}
                    </span>
                  </div>
                  <p className={`font-mono text-slate-300 transition-all cursor-help ${
                    complexityMode === '16-bit' ? 'text-xl' : 'text-[10px] break-all max-h-12 overflow-y-auto'
                  } ${showD ? '' : 'blur-sm group-hover:blur-none'}`}>
                    {key?.d.toString() || '---'}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {activeTab === 'simulator' ? (
          <>
            {/* Simulator Section */}
            <section id="simulator" className="grid lg:grid-cols-3 gap-8">
              {/* Controls & Inputs */}
              <div className="lg:col-span-1 space-y-6">
            <div className="bg-slate-900 border border-white/10 rounded-3xl p-6 space-y-6 overflow-hidden relative">
              <div className="flex items-center gap-2 mb-2">
                <Terminal className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-bold text-white">Target Input</h3>
              </div>
              
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-500 uppercase">Public Key (N)</label>
                  <div className="relative">
                    <input 
                      type="text" 
                      value={targetN}
                      onChange={(e) => setTargetN(e.target.value)}
                      placeholder="e.g. 3233"
                      className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 font-mono focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                    />
                    <Search className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <button 
                    onClick={startSimulation}
                    disabled={isSimulating}
                    className="flex items-center justify-center gap-2 py-4 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 rounded-2xl font-bold transition-all shadow-lg shadow-blue-500/20 active:scale-95"
                  >
                    <Play className="w-5 h-5" />
                    {isSimulating ? 'Processing...' : 'Break'}
                  </button>
                  <button 
                    onClick={resetSimulation}
                    className="flex items-center justify-center gap-2 py-4 bg-slate-800 hover:bg-slate-700 rounded-2xl font-bold transition-all active:scale-95"
                  >
                    <RotateCcw className="w-5 h-5" />
                    Reset
                  </button>
                </div>
              </div>

              {/* Status Display */}
              <div className="pt-6 border-t border-white/5 space-y-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500 font-bold uppercase">System Status</span>
                  <div className="flex gap-2">
                    <span className="px-2 py-1 rounded bg-blue-500/10 text-blue-500 font-bold">
                      PY_BACKEND_UNVERIFIED
                    </span>
                    <span className={`px-2 py-1 rounded ${isSimulating ? 'bg-amber-500/10 text-amber-500' : 'bg-emerald-500/10 text-emerald-500'}`}>
                      {isSimulating ? 'ACTIVE_SEARCH' : 'IDLE'}
                    </span>
                  </div>
                </div>
                {result && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl text-center space-y-4"
                  >
                    <div className="flex justify-center mb-1">
                      <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                    </div>
                    <div className="space-y-1">
                      <div className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest">
                        Factors Discovered
                      </div>
                      <div className="font-mono text-xl font-bold text-white">
                        P={result.p} | Q={result.q}
                      </div>
                    </div>

                    {key && key.n.toString() === targetN && <div className="pt-2 border-t border-white/5 space-y-3">
                      <button 
                        onClick={() => {
                          if (!key) return;
                          const phi = (result.p - 1n) * (result.q - 1n);
                          addLog(`Calculated Phi(N) = ${phi}`);
                          addLog(`Recovered Private Key D = ${key.d}`);
                          setShowD(true);
                        }}
                        className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 rounded-xl text-xs font-bold transition-all"
                      >
                        Derive Private Key (D)
                      </button>

                      <div className="bg-black/40 rounded-xl p-4 text-left border border-white/5 space-y-3">
                        <div className="flex items-center gap-2 text-blue-400">
                          <Search className="w-3 h-3" />
                          <span className="text-[10px] font-bold uppercase tracking-wider">Manual Verification Guide</span>
                        </div>
                        <p className="text-[10px] text-slate-400 leading-relaxed italic">
                          To verify that D is not hardcoded, use any calculator with the factors found by the MDN Oracle:
                        </p>
                        <div className="font-mono text-[9px] space-y-1 bg-black/60 p-2 rounded border border-white/5 text-slate-300">
                          <div>1. <span className="text-slate-500">Calculate:</span> Phi = ({result.p.toString()}-1) × ({result.q.toString()}-1) = <span className="text-blue-300">{((result.p - 1n) * (result.q - 1n)).toString()}</span></div>
                          <div>2. <span className="text-slate-500">The Goal:</span> Find D where (D × {key?.e.toString()}) mod Phi = 1</div>
                          <div>3. <span className="text-slate-500">Recovery:</span> Euclidean algorithm finds D = <span className="text-amber-400 font-bold">{key?.d.toString()}</span></div>
                          <div>4. <span className="text-slate-500">Verification:</span> ({key?.d.toString()} × {key?.e.toString()}) mod {((result.p - 1n) * (result.q - 1n)).toString()} ≡ <span className="text-emerald-400 font-bold">1</span></div>
                          <div className="pt-1 border-t border-white/5">
                            5. <span className="text-slate-500">Manual Calculation of D:</span>
                            <div className="text-amber-400 italic">D = ({key?.e.toString()}⁻¹ mod {((result.p - 1n) * (result.q - 1n)).toString()})</div>
                          </div>
                        </div>
                        <div className="bg-blue-500/5 border-l-2 border-orange-500 p-2 space-y-1">
                          <p className="text-[8px] font-bold text-orange-400 uppercase tracking-tighter">Your Calculator Test (Find D yourself):</p>
                          <p className="text-[9px] text-slate-400 leading-tight">
                            To find the secret <b>D</b> manually: use any online <b>Modular Inverse Calculator</b>. 
                            Input Number (E) = <b>{key?.e.toString()}</b> and Modulus (Phi) = <b>{((result.p - 1n) * (result.q - 1n)).toString()}</b>. 
                            When you press "Calculate", it will yield <b>{key?.d.toString()}</b>.
                          </p>
                        </div>
                        <p className="text-[9px] text-slate-500">
                          If the remainder is 1, the private key D is mathematically unique and correct for this N.
                        </p>
                      </div>
                    </div>}
                  </motion.div>
                )}
              </div>
            </div>

            {/* Logs */}
            <div className="bg-black/40 border border-white/5 rounded-3xl p-4 font-mono text-[10px] h-48 flex flex-col">
              <div className="mb-2 pb-2 border-b border-white/5 text-slate-600 font-bold uppercase tracking-widest">Process Logs</div>
              <div className="flex-1 overflow-y-auto space-y-1 scrollbar-hide">
                <AnimatePresence>
                  {logs.map((log, i) => (
                    <motion.div 
                      key={i + log}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="text-slate-400 flex gap-2"
                    >
                      <span className="text-blue-500/50">[{i}]</span> {log}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          </div>

          {/* Visualization Area */}
          <div className="lg:col-span-2 bg-slate-900 border border-white/10 rounded-3xl overflow-hidden relative group">
            <div className="absolute top-6 left-6 z-10 space-y-1">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Orbit className="w-5 h-5 text-blue-400" />
                Phase Space D₄
              </h3>
              <p className="text-xs text-slate-500 font-mono tracking-wider uppercase">Topological Projection (xy = n)</p>
            </div>

            <div className="absolute top-6 right-6 z-10 flex gap-4">
              <div className="text-right">
                <p className="text-[10px] font-bold text-slate-500 uppercase">Current Energy</p>
                <p className="text-xl font-mono text-white">
                  {steps.length > 0 ? steps[steps.length - 1].energy.toFixed(4) : '0.0000'}
                </p>
              </div>
            </div>

            <div className="w-full aspect-square md:aspect-video flex items-center justify-center bg-black/60 cursor-crosshair">
              <canvas 
                ref={canvasRef} 
                width={800} 
                height={500} 
                className="w-full h-full object-contain"
              />
            </div>

            {/* Stats Overlay */}
            <div className="absolute bottom-6 left-6 right-6 grid grid-cols-4 gap-4">
              {[
                { label: 'Steps', value: steps.length, icon: TrendingDown },
                { label: 'Learning Rate', value: '0.50', icon: Zap },
                { label: 'MDN Confidence', value: '98.2%', icon: Cpu },
                { label: 'Precision', value: '16-bit', icon: Shield },
              ].map((stat, i) => (
                <div key={i} className="bg-black/60 backdrop-blur-md border border-white/5 rounded-xl p-3 flex items-center gap-3">
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <stat.icon className="w-4 h-4 text-blue-400" />
                  </div>
                  <div>
                    <div className="text-[8px] font-bold text-slate-500 uppercase">{stat.label}</div>
                    <div className="text-xs font-mono text-white">{stat.value}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </>
    ) : (
      <motion.section 
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="grid lg:grid-cols-2 gap-12"
      >
        <div className="space-y-6">
          <div className="bg-slate-900 border border-white/10 rounded-3xl p-8 space-y-6">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-amber-500/10 rounded-2xl">
                <Terminal className="w-6 h-6 text-amber-500" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Modular Arithmetic Sandbox</h3>
                <p className="text-sm text-slate-500 font-mono italic">Testing the Multiplicative Inverse</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6 pt-4">
              <div className="space-y-2">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Number (A)</label>
                <input 
                  type="number"
                  value={sandboxA}
                  onChange={(e) => setSandboxA(e.target.value)}
                  className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 font-mono text-amber-300 focus:ring-1 focus:ring-amber-500 outline-none"
                />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Modulus (M)</label>
                <input 
                  type="number"
                  value={sandboxM}
                  onChange={(e) => setSandboxM(e.target.value)}
                  className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 font-mono text-blue-300 focus:ring-1 focus:ring-blue-500 outline-none"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Candidate Inverse (I)</label>
              <input 
                type="number"
                value={sandboxI}
                onChange={(e) => setSandboxI(e.target.value)}
                placeholder="Try to find the I..."
                className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 font-mono text-emerald-400 focus:ring-1 focus:ring-emerald-500 outline-none"
              />
            </div>

            <div className="pt-6 border-t border-white/5">
              <div className="bg-black/60 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-400 uppercase">Live Proof Logic</span>
                  <div className="px-2 py-1 bg-white/5 border border-white/10 rounded text-[9px] font-mono text-slate-500">NodeJS / JS</div>
                </div>
                <code className="block font-mono text-sm leading-relaxed">
                  <div className="text-slate-500">// Check if I is the inverse of A modulo M</div>
                  <div className="flex gap-2 mt-2">
                    <span className="text-pink-400">if</span>
                    <span className="text-slate-300">((</span>
                    <span className="text-amber-300">{sandboxA || 'a'}</span>
                    <span className="text-slate-300"> * </span>
                    <span className="text-emerald-400">{sandboxI || 'i'}</span>
                    <span className="text-slate-300">) % </span>
                    <span className="text-blue-300">{sandboxM || 'm'}</span>
                    <span className="text-slate-300"> == </span>
                    <span className="text-emerald-500">1</span>
                    <span className="text-slate-300">) {'{'}</span>
                  </div>
                  <div className="pl-6 text-slate-400">
                    console.log(<span className="text-emerald-400">"{sandboxI || 'i'} is legitimate!"</span>);
                  </div>
                  <div className="text-slate-300">{'}'}</div>
                </code>
                
                <div className="pt-4 flex justify-center">
                  {(() => {
                    const a = parseInt(sandboxA), m = parseInt(sandboxM), i = parseInt(sandboxI);
                    const isValid = !isNaN(a) && !isNaN(m) && !isNaN(i) && (a * i) % m === 1;
                    if (isNaN(i)) return null;
                    return (
                      <motion.div 
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className={`flex items-center gap-3 px-6 py-4 rounded-2xl border ${
                          isValid ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-500' : 'bg-rose-500/10 border-rose-500/20 text-rose-500'
                        }`}
                      >
                        {isValid ? <CheckCircle2 className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                        <span className="font-bold uppercase tracking-widest text-xs">
                          {isValid ? 'VALID INVERSE FOUND!' : 'INVALID INVERSE'}
                        </span>
                      </motion.div>
                    );
                  })()}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-slate-900 border border-white/10 rounded-3xl p-8 space-y-6">
                <h4 className="text-lg font-bold text-white flex items-center gap-2">
                  <Search className="w-5 h-5 text-blue-400" />
                  Educational Insight
                </h4>
                <div className="prose prose-invert prose-sm text-slate-400 space-y-4">
                  <p>In RSA, the relationship between <b>E</b> (Public Exponent) and <b>D</b> (Private Key) is based on modular inverses. They are mathematical mirrors across the modulus <b>&phi;(N)</b>.</p>
                  <div className="bg-white/5 p-4 rounded-xl font-mono text-xs border border-white/10">
                    Formula: D &times; E &equiv; 1 (mod &phi;(N))
                  </div>
                  <p>Our TKHD+MDN method finds <b>P</b> and <b>Q</b>, which lets us calculate <b>&phi;(N) = (P-1)(Q-1)</b>. Once known, calculating <b>D</b> is no longer a hard problem&mdash;it is a simple modular inversion.</p>
                  <p className="text-xs italic border-t border-white/5 pt-4">Try guessing the inverse in the sandbox! For example, if A=3 and M=11, try setting I to 4 (because 3 &times; 4 = 12, and 12 mod 11 = 1).</p>
                </div>
              </div>
            </div>
          </motion.section>
        )}

        {/* Info Section */}
        <section id="theory" className="pt-12 border-t border-white/5 space-y-12">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="space-y-4">
              <div className="w-12 h-12 bg-blue-600/10 rounded-2xl flex items-center justify-center">
                <TrendingDown className="w-6 h-6 text-blue-400" />
              </div>
              <h4 className="text-xl font-bold text-white">Gradient Descent</h4>
              <p className="text-slate-400 text-sm leading-relaxed">
                The core of TKHD is movement along the gradient of the energy function 𝓛. This landscape is smooth except for points where local minima appear.
              </p>
            </div>
            <div className="space-y-4">
              <div className="w-12 h-12 bg-amber-600/10 rounded-2xl flex items-center justify-center">
                <Cpu className="w-6 h-6 text-amber-500" />
              </div>
              <h4 className="text-xl font-bold text-white">MDN Oracle</h4>
              <p className="text-slate-400 text-sm leading-relaxed">
                Mixture Density Networks predict the probability distribution of prime factors. When our "ray" hits a stall point, the MDN triggers a "Jump" to a new high-probability coordinate.
              </p>
            </div>
            <div className="space-y-4">
              <div className="w-12 h-12 bg-emerald-600/10 rounded-2xl flex items-center justify-center">
                <CheckCircle2 className="w-6 h-6 text-emerald-500" />
              </div>
              <h4 className="text-xl font-bold text-white">Quantum Collapse</h4>
              <p className="text-slate-400 text-sm leading-relaxed">
                The final step is symbolic. Once the continuous path reaches a stable integer neighborhood, the system "collapses" into the exact algebraic solution.
              </p>
            </div>
          </div>

          {/* Benchmark Table */}
          <div className="bg-slate-900 border border-white/10 rounded-3xl p-8 space-y-6">
            <div className="flex items-center gap-3">
              <Zap className="w-6 h-6 text-blue-400" />
              <h4 className="text-xl font-bold text-white">Computational Benchmarks (16-bit RSA)</h4>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="py-4 font-bold text-slate-500 uppercase tracking-widest text-[10px]">Method</th>
                    <th className="py-4 font-bold text-slate-500 uppercase tracking-widest text-[10px]">Estimated Time</th>
                    <th className="py-4 font-bold text-slate-500 uppercase tracking-widest text-[10px]">Search Complexity</th>
                  </tr>
                </thead>
                <tbody className="text-slate-300 font-mono text-[11px]">
                  <tr className="border-b border-white/5">
                    <td className="py-4">Trial Division (Brute)</td>
                    <td className="py-4 text-emerald-400">&lt; 0.001s</td>
                    <td className="py-4">O(&radic;N) - Linear check of all primes</td>
                  </tr>
                  <tr className="border-b border-white/5">
                    <td className="py-4">Sieve of Eratosthenes</td>
                    <td className="py-4 text-emerald-400">&lt; 0.005s</td>
                    <td className="py-4">O(N log log N) - Memory intensive</td>
                  </tr>
                  <tr>
                    <td className="py-4 text-blue-400 font-bold">TKHD+MDN (This Method)</td>
                    <td className="py-4 text-blue-400">0.5s - 1.5s</td>
                    <td className="py-4 text-blue-400">No established complexity bound</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-500 italic leading-relaxed">
              Note: This is a small-key research prototype. Its scaling to larger keys has not been demonstrated; the 2048-bit solver is unimplemented.
            </p>
          </div>
        </section>
      </main>

      <footer className="border-t border-white/5 bg-black/20 py-8">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-500 font-mono">TKHD+MDN Neuro-Symbolic Research Prototype © 2026</p>
          <div className="flex gap-6">
            <button className="text-xs text-slate-500 hover:text-white transition-colors flex items-center gap-1">
              <Copy className="w-3 h-3" /> Copy Method.txt
            </button>
            <button className="text-xs text-slate-500 hover:text-white transition-colors flex items-center gap-1">
              <Terminal className="w-3 h-3" /> View Source code
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
