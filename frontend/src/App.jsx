import { useCallback, useEffect, useState } from 'react'
import { getJSON } from './api'
import FaceToSketch from './workspaces/FaceToSketch'
import HardRouted from './workspaces/HardRouted'
import SoftMoE from './workspaces/SoftMoE'
import SystemStatus from './workspaces/SystemStatus'
import Universal from './workspaces/Universal'

const WORKSPACES = [
  {
    id: 'universal',
    task: 'Task 1',
    name: 'Universal Restoration',
    blurb: 'One denoising autoencoder for clean, salt-and-pepper, blurred and occluded inputs — it is never told the corruption type.',
    icon: '◎',
    Component: Universal,
  },
  {
    id: 'hard_routing',
    task: 'Task 2',
    name: 'Hard-Routed Restoration',
    blurb: 'A corruption classifier picks exactly one specialist autoencoder; clean inputs take the identity bypass.',
    icon: '⑂',
    Component: HardRouted,
  },
  {
    id: 'soft_moe',
    task: 'Task 3',
    name: 'Soft Mixture-of-Experts Restoration',
    blurb: 'A gating network blends the identity branch and all three experts with continuous weights.',
    icon: '◈',
    Component: SoftMoE,
  },
  {
    id: 'face2sketch',
    task: 'Task 4',
    name: 'Face-to-Sketch Generator',
    blurb: 'Style-conditioned U-Net generator from a conditional GAN trained on FS2K.',
    icon: '✎',
    Component: FaceToSketch,
  },
]

export default function App() {
  const [active, setActive] = useState(() => location.hash.slice(1) || 'universal')
  const [health, setHealth] = useState(null)
  const [healthErr, setHealthErr] = useState(null)
  const [navOpen, setNavOpen] = useState(false)

  const refresh = useCallback(() => {
    getJSON('/api/health')
      .then((h) => {
        setHealth(h)
        setHealthErr(null)
      })
      .catch((e) => setHealthErr(e.message))
  }, [])

  useEffect(() => {
    const onHash = () => setActive(location.hash.slice(1) || 'universal')
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 15000)
    return () => clearInterval(id)
  }, [refresh])

  function go(id) {
    setActive(id)
    setNavOpen(false)
    history.replaceState(null, '', `#${id}`)
  }

  const current = WORKSPACES.find((w) => w.id === active)
  const online = health && !healthErr

  return (
    <div className="min-h-screen lg:flex">
      {/* sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-72 transform border-r border-slate-200 bg-white transition lg:sticky lg:top-0 lg:h-screen lg:shrink-0 lg:translate-x-0 ${
          navOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-16 items-center gap-3 border-b border-slate-100 px-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-lg font-bold text-white">R</div>
          <div>
            <div className="text-sm font-semibold text-slate-900">RestoreLab</div>
            <div className="text-[11px] text-slate-500">Generative AI · Assignment 1</div>
          </div>
        </div>
        <nav className="space-y-1 p-3">
          <p className="px-3 pt-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Workspaces</p>
          {WORKSPACES.map((w) => {
            const ready = health?.tasks?.[w.id]
            return (
              <button
                key={w.id}
                onClick={() => go(w.id)}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition ${
                  active === w.id ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-50'
                }`}
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white text-base shadow-sm ring-1 ring-slate-200">
                  {w.icon}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block text-[11px] font-medium text-slate-400">{w.task}</span>
                  <span className="block truncate text-sm font-medium">{w.name}</span>
                </span>
                {health && (
                  <span
                    title={ready ? 'model ready' : 'model file missing'}
                    className={`h-2 w-2 shrink-0 rounded-full ${ready ? 'bg-emerald-500' : 'bg-slate-300'}`}
                  />
                )}
              </button>
            )
          })}
          <p className="px-3 pt-4 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">System</p>
          <button
            onClick={() => go('status')}
            className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition ${
              active === 'status' ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-50'
            }`}
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-white shadow-sm ring-1 ring-slate-200">⚙</span>
            Health & models
          </button>
        </nav>
      </aside>
      {navOpen && <div className="fixed inset-0 z-20 bg-slate-900/30 lg:hidden" onClick={() => setNavOpen(false)} />}

      {/* main */}
      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b border-slate-200 bg-white/80 px-4 backdrop-blur sm:px-8">
          <button className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 lg:hidden" onClick={() => setNavOpen(true)}>
            ☰
          </button>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-semibold text-slate-900">{current ? current.name : 'Health & models'}</h1>
          </div>
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${
              online ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${online ? 'bg-emerald-500' : 'bg-red-500'}`} />
            {online ? `API online · ORT ${health.onnxruntime}` : healthErr ? 'API offline' : 'Connecting…'}
          </span>
        </header>

        <main className="mx-auto max-w-[1400px] p-4 sm:p-8">
          {current && (
            <div className="mb-6">
              <span className="text-xs font-semibold uppercase tracking-wider text-brand-600">{current.task}</span>
              <p className="mt-1 max-w-3xl text-sm text-slate-600">{current.blurb}</p>
              {health && !health.tasks[current.id] && (
                <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-800">
                  The ONNX model for this workspace is not in <code className="font-mono text-xs">{health.model_dir}</code> yet —
                  requests will fail until it is exported.
                </div>
              )}
            </div>
          )}
          {/* keep every workspace mounted so inputs/results survive tab switches */}
          {WORKSPACES.map(({ id, Component }) => (
            <div key={id} hidden={active !== id}>
              <Component />
            </div>
          ))}
          {active === 'status' && <SystemStatus health={health} error={healthErr} onRefresh={refresh} />}
        </main>
      </div>
    </div>
  )
}
