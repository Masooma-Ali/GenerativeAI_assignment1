import { Card, KeyValue } from '../components/ui'

const TASK_NAMES = {
  universal: 'Universal Restoration',
  hard_routing: 'Hard-Routed Restoration',
  soft_moe: 'Soft Mixture-of-Experts Restoration',
  face2sketch: 'Face-to-Sketch Generator',
}

function Dot({ ok }) {
  return <span className={`inline-block h-2 w-2 rounded-full ${ok ? 'bg-emerald-500' : 'bg-red-400'}`} />
}

export default function SystemStatus({ health, error, onRefresh }) {
  if (error)
    return (
      <Card title="Backend unreachable">
        <p className="text-sm text-red-600">{error}</p>
        <p className="mt-2 text-sm text-slate-500">
          Start it from the repository root with <code className="rounded bg-slate-100 px-1">uvicorn backend.app.main:app --port 8000</code>
        </p>
      </Card>
    )
  if (!health) return <p className="text-sm text-slate-500">Loading…</p>

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Server"
          actions={
            <button onClick={onRefresh} className="text-xs font-medium text-brand-600 hover:underline">
              Refresh
            </button>
          }
        >
          <KeyValue
            rows={[
              ['Status', health.status],
              ['Uptime', `${health.uptime_s} s`],
              ['Python', health.python],
              ['ONNX Runtime', health.onnxruntime],
              ['Providers', health.providers.join(', ')],
              ['Model directory', <span className="font-mono text-xs break-all">{health.model_dir}</span>],
            ]}
          />
        </Card>
        <Card title="Workspaces">
          <ul className="space-y-2 text-sm">
            {Object.entries(health.tasks).map(([k, ok]) => (
              <li key={k} className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Dot ok={ok} /> {TASK_NAMES[k]}
                </span>
                <span className={`text-xs font-medium ${ok ? 'text-emerald-600' : 'text-red-500'}`}>
                  {ok ? 'ready' : 'model missing'}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Card title="ONNX models" subtitle="Export reports compare ONNX Runtime output with the PyTorch model">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-slate-500">
              <tr className="border-b border-slate-200">
                <th className="py-2 pr-4">File</th>
                <th className="py-2 pr-4">Status</th>
                <th className="py-2 pr-4">Size</th>
                <th className="py-2 pr-4">Max |ONNX − PyTorch|</th>
                <th className="py-2 pr-4">Consistent</th>
                <th className="py-2">Load</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(health.models).map(([k, m]) => {
                const r = m.export_report || {}
                const diff = r.max_abs_diff ?? r.max_abs_diff_restored
                return (
                  <tr key={k} className="border-b border-slate-100">
                    <td className="py-2 pr-4 font-mono text-xs">{m.file}</td>
                    <td className="py-2 pr-4">
                      <span className="flex items-center gap-2">
                        <Dot ok={m.available} />
                        {m.available ? (m.loaded ? 'loaded' : 'on disk') : 'missing'}
                      </span>
                    </td>
                    <td className="py-2 pr-4">{m.size_mb ? `${m.size_mb} MB` : '—'}</td>
                    <td className="py-2 pr-4 font-mono text-xs">{diff !== undefined ? diff.toExponential(2) : '—'}</td>
                    <td className="py-2 pr-4">{r.consistent === undefined ? '—' : r.consistent ? '✓' : '✗'}</td>
                    <td className="py-2">{m.load_ms ? `${m.load_ms} ms` : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
