import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const api = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')
type Result = { prediction: number; confidence?: number; model_version?: string; device?: string; latency_ms: number }

function App() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const drawing = useRef(false)
  const [hasInk, setHasInk] = useState(false)
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [loading, setLoading] = useState(false)
  const [device, setDevice] = useState<'cpu' | 'cuda'>('cpu')
  const [cudaAvailable, setCudaAvailable] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [timings, setTimings] = useState<Partial<Record<'cpu' | 'cuda', number>>>({})
  const [error, setError] = useState('')

  function clear() {
    const element = canvas.current
    if (!element) return
    const ctx = element.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, element.width, element.height)
    setHasInk(false)
    setResult(null)
    setTimings({})
    setError('')
  }

  useEffect(() => {
    clear()
    const controller = new AbortController()
    fetch(`${api}/health`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error(); return response.json() })
      .then(data => { setCudaAvailable(data.cuda_available === true); setStatus('online') })
      .catch(() => { if (!controller.signal.aborted) setStatus('offline') })
    return () => controller.abort()
  }, [])

  function point(event: PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    return { x: (event.clientX - rect.left) * event.currentTarget.width / rect.width, y: (event.clientY - rect.top) * event.currentTarget.height / rect.height }
  }
  function start(event: PointerEvent<HTMLCanvasElement>) {
    if (loading || (event.pointerType === 'mouse' && event.button !== 0)) return
    event.currentTarget.setPointerCapture(event.pointerId)
    drawing.current = true
    const ctx = event.currentTarget.getContext('2d')!
    const { x, y } = point(event)
    ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineWidth = 22; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
    ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.fill()
    ctx.beginPath(); ctx.moveTo(x, y)
    setHasInk(true); setResult(null); setTimings({}); setError('')
  }
  function move(event: PointerEvent<HTMLCanvasElement>) {
    if (!drawing.current) return
    const ctx = event.currentTarget.getContext('2d')!
    const { x, y } = point(event)
    ctx.lineTo(x, y); ctx.stroke()
  }
  function stop() { drawing.current = false }

  async function submit() {
    const element = canvas.current
    if (!element || !hasInk) return
    setLoading(true); setError(''); setResult(null)
    try {
      const blob = await new Promise<Blob>((resolve, reject) => element.toBlob(value => value ? resolve(value) : reject(new Error('Could not create the image.')), 'image/png'))
      const form = new FormData(); form.append('file', blob, 'digit.png'); form.append('device', device)
      const response = await fetch(`${api}/predict`, { method: 'POST', body: form })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(typeof body?.detail === 'string' ? body.detail : `Prediction failed (${response.status}).`)
      }
      const prediction = await response.json() as Result
      setResult(prediction); setTimings(previous => ({ ...previous, [device]: prediction.latency_ms })); setStatus('online')
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not reach the backend.') }
    finally { setLoading(false) }
  }

  return <main className="shell">
    <header><span className="eyebrow">VISION INFERENCE <span className="dot">/</span> MNIST</span><span className={`status ${status}`}><i />{status === 'checking' ? 'Checking connection' : status === 'online' ? 'Backend connected' : 'Backend unavailable'}</span></header>
    <section className="intro"><p className="kicker">HANDWRITTEN DIGIT RECOGNITION</p><h1>Draw a digit.<br /><em>See what the model sees.</em></h1><p>Write a single digit from 0 to 9 in the canvas, then ask the model for its prediction.</p></section>
    <section className="workspace"><div className="drawing-panel"><div className="panel-heading"><div><span className="step">01 / INPUT</span><h2>Your drawing</h2></div><span className="hint">Use mouse or touch</span></div><canvas ref={canvas} width={280} height={280} aria-label="Digit drawing canvas" onPointerDown={start} onPointerMove={move} onPointerUp={stop} onPointerCancel={stop} onLostPointerCapture={stop} /><label className="device">Compute device <select value={device} disabled={loading} onChange={event => setDevice(event.target.value as 'cpu' | 'cuda')}><option value="cpu">CPU</option><option value="cuda" disabled={!cudaAvailable}>GPU (CUDA){!cudaAvailable ? ' — unavailable' : ''}</option></select></label><div className="actions"><button className="secondary" onClick={clear} disabled={loading}>Clear</button><button className="primary" onClick={submit} disabled={!hasInk || loading}>{loading ? 'Predicting…' : 'Predict digit'} <span>↗</span></button></div></div><div className="result-panel"><span className="step">02 / RESULT</span><h2>Prediction</h2><div className="result" aria-live="polite">{loading ? <><span className="result-label">ANALYZING DRAWING</span><span className="placeholder">···</span></> : result ? <><span className="result-label">THE MODEL PREDICTS</span><strong>{result.prediction}</strong>{typeof result.confidence === 'number' && <p className="confidence">{(result.confidence * 100).toFixed(1)}% <span>confidence</span></p>}{result.model_version && <small>Model version {result.model_version} · {result.device === 'cuda' ? 'GPU' : 'CPU'}</small>}</> : <><span className="placeholder">?</span><p>Your prediction will<br />appear here.</p></>}</div><div className="timings" aria-live="polite"><div className="timings-heading"><span className="step">INFERENCE TIME</span><span>Latest run on this drawing</span></div><div className="timing-grid">{(['cpu', 'cuda'] as const).map(item => <div className="timing" key={item}><span>{item === 'cpu' ? 'CPU' : 'GPU'}</span><strong>{timings[item] === undefined ? '—' : `${timings[item].toFixed(2)} ms`}</strong></div>)}</div><p>Includes preprocessing and prediction; excludes model transfer. First runs may be slower.</p></div>{error && <p className="error" role="alert">{error}</p>}</div></section><footer><span>POWERED BY YOUR REGISTERED MODEL</span><span>DRAW · PREDICT · EXPLORE</span></footer>
  </main>
}

createRoot(document.getElementById('root')!).render(<App />)
