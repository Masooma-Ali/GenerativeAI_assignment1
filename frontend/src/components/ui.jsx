export function Card({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`rounded-2xl border border-slate-200 bg-white shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-slate-100 px-5 py-3.5">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

export function Button({ children, variant = 'primary', className = '', ...props }) {
  const styles = {
    primary: 'bg-brand-600 text-white hover:bg-brand-700 disabled:bg-slate-300',
    secondary: 'bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 disabled:text-slate-400',
    ghost: 'text-slate-600 hover:bg-slate-100',
  }
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed ${styles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}

export function Spinner() {
  return <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
}

export function ErrorBanner({ error }) {
  if (!error) return null
  return (
    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <span className="font-semibold">Error: </span>
      {error}
    </div>
  )
}

export function Label({ children }) {
  return <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-slate-500">{children}</label>
}

export function Select({ value, onChange, options, ...props }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-100 focus:outline-none"
      {...props}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function NumberInput({ value, onChange, ...props }) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-100 focus:outline-none"
      {...props}
    />
  )
}

export function Segmented({ value, onChange, options }) {
  return (
    <div className="inline-flex flex-wrap rounded-lg bg-slate-100 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
            value === o.value ? 'bg-white text-brand-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a')
  a.href = dataUrl
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
}

export function ImageCard({ title, src, caption, downloadName, highlight = false }) {
  return (
    <figure
      className={`flex flex-col overflow-hidden rounded-xl border bg-white ${
        highlight ? 'border-brand-500 ring-2 ring-brand-100' : 'border-slate-200'
      }`}
    >
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <figcaption className="truncate text-xs font-semibold text-slate-700" title={title}>
          {title}
        </figcaption>
        {downloadName && src && (
          <button
            onClick={() => downloadDataUrl(src, downloadName)}
            className="shrink-0 rounded-md px-1.5 py-0.5 text-xs font-semibold text-brand-600 hover:bg-brand-50"
            title={`Download ${downloadName}`}
          >
            ↓ PNG
          </button>
        )}
      </div>
      <div className="aspect-square bg-slate-100">
        {src ? (
          <img src={src} alt={title} className="pixelated h-full w-full object-contain" />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-400">—</div>
        )}
      </div>
      {caption && <div className="px-3 py-2 text-[11px] text-slate-500">{caption}</div>}
    </figure>
  )
}

const BAR_COLORS = {
  clean: 'bg-emerald-500',
  salt_pepper: 'bg-amber-500',
  blur: 'bg-sky-500',
  occlusion: 'bg-fuchsia-500',
}

/** Horizontal bars for classifier probabilities or MoE weights. */
export function WeightBars({ values, labels, highlight, trueLabel }) {
  const entries = Object.entries(values)
  const max = Math.max(...entries.map(([, v]) => v))
  return (
    <div className="space-y-3">
      {entries.map(([k, v]) => {
        const isTop = highlight ? k === highlight : v === max
        return (
          <div key={k}>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className={`font-medium ${isTop ? 'text-slate-900' : 'text-slate-600'}`}>
                {labels[k] || k}
                {isTop && <span className="ml-2 rounded bg-brand-50 px-1.5 py-0.5 text-[10px] font-semibold text-brand-700">TOP</span>}
                {trueLabel === k && (
                  <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600">TRUE</span>
                )}
              </span>
              <span className="font-mono tabular-nums text-slate-700">{(v * 100).toFixed(2)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full transition-all duration-500 ${BAR_COLORS[k] || 'bg-brand-500'} ${isTop ? '' : 'opacity-60'}`}
                style={{ width: `${Math.max(v * 100, 0.5)}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function KeyValue({ rows }) {
  return (
    <dl className="divide-y divide-slate-100 text-sm">
      {rows
        .filter((r) => r && r[1] !== undefined && r[1] !== null)
        .map(([k, v]) => (
          <div key={k} className="flex items-center justify-between gap-4 py-1.5">
            <dt className="text-slate-500">{k}</dt>
            <dd className="text-right font-medium text-slate-800">{v}</dd>
          </div>
        ))}
    </dl>
  )
}

export function Stat({ label, value, unit, tone = 'slate' }) {
  const tones = { slate: 'text-slate-900', green: 'text-emerald-600', red: 'text-red-600', brand: 'text-brand-700' }
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${tones[tone]}`}>
        {value}
        {unit && <span className="ml-1 text-xs font-medium text-slate-500">{unit}</span>}
      </div>
    </div>
  )
}
