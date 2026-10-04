import { Label, NumberInput, Select } from './ui'

export const DEFAULT_CORRUPTION = {
  corruption: 'salt_pepper',
  severity: 'medium',
  seed: '42',
  p: '0.08',
  kernel: '5',
  sigma: '1.5',
  n_rects: '2',
  coverage: '0.2',
}

const TYPES = [
  { value: 'none', label: 'None — my upload is already corrupted' },
  { value: 'clean', label: 'Clean (no corruption)' },
  { value: 'salt_pepper', label: 'Salt-and-pepper noise' },
  { value: 'blur', label: 'Gaussian blur' },
  { value: 'occlusion', label: 'Rectangular occlusion' },
]

const LEVELS = {
  salt_pepper: ['p = 0.03', 'p = 0.08', 'p = 0.15'],
  blur: ['k=3, σ=0.7', 'k=5, σ=1.5', 'k=7, σ=2.5'],
  occlusion: ['1 rect, ~10%', '2 rects, ~20%', '3 rects, ~35%'],
}

/** Fields sent to the backend for a given control state. */
export function corruptionFields(c) {
  if (c.corruption === 'none' || c.corruption === 'clean') return { corruption: c.corruption, seed: c.seed }
  const f = { corruption: c.corruption, severity: c.severity, seed: c.seed }
  if (c.severity === 'custom') {
    if (c.corruption === 'salt_pepper') f.p = c.p
    if (c.corruption === 'blur') Object.assign(f, { kernel: c.kernel, sigma: c.sigma })
    if (c.corruption === 'occlusion') Object.assign(f, { n_rects: c.n_rects, coverage: c.coverage })
  }
  return f
}

export default function CorruptionControls({ value, onChange }) {
  const set = (k) => (v) => onChange({ ...value, [k]: v })
  const t = value.corruption
  const hasSeverity = !['none', 'clean'].includes(t)
  const levels = LEVELS[t] || []

  return (
    <div className="space-y-4">
      <div>
        <Label>Runtime corruption</Label>
        <Select value={t} onChange={set('corruption')} options={TYPES} />
        <p className="mt-1.5 text-xs text-slate-500">
          {t === 'none'
            ? 'The upload is fed to the model as-is. No clean target, so PSNR/SSIM are not computed.'
            : 'Your upload is treated as the clean target; the corruption is applied on the server before inference.'}
        </p>
      </div>

      {hasSeverity && (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Severity</Label>
            <Select
              value={value.severity}
              onChange={set('severity')}
              options={[
                { value: 'low', label: `Low (${levels[0]})` },
                { value: 'medium', label: `Medium (${levels[1]})` },
                { value: 'high', label: `High (${levels[2]})` },
                { value: 'random', label: 'Random (training range)' },
                { value: 'custom', label: 'Custom…' },
              ]}
            />
          </div>
          <div>
            <Label>Seed</Label>
            <NumberInput value={value.seed} onChange={set('seed')} min={0} />
          </div>
        </div>
      )}

      {hasSeverity && value.severity === 'custom' && (
        <div className="grid grid-cols-2 gap-3 rounded-xl bg-slate-50 p-3">
          {t === 'salt_pepper' && (
            <div className="col-span-2">
              <Label>Probability p ({value.p})</Label>
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={value.p}
                onChange={(e) => set('p')(e.target.value)}
                className="w-full accent-brand-600"
              />
            </div>
          )}
          {t === 'blur' && (
            <>
              <div>
                <Label>Kernel</Label>
                <Select
                  value={value.kernel}
                  onChange={set('kernel')}
                  options={[3, 5, 7, 9, 11].map((k) => ({ value: String(k), label: `${k}×${k}` }))}
                />
              </div>
              <div>
                <Label>Sigma σ</Label>
                <NumberInput value={value.sigma} onChange={set('sigma')} min={0.1} max={10} step={0.1} />
              </div>
            </>
          )}
          {t === 'occlusion' && (
            <>
              <div>
                <Label>Rectangles</Label>
                <Select
                  value={value.n_rects}
                  onChange={set('n_rects')}
                  options={[1, 2, 3].map((k) => ({ value: String(k), label: String(k) }))}
                />
              </div>
              <div>
                <Label>Coverage ({Math.round(value.coverage * 100)}%)</Label>
                <input
                  type="range"
                  min="0.05"
                  max="0.6"
                  step="0.01"
                  value={value.coverage}
                  onChange={(e) => set('coverage')(e.target.value)}
                  className="mt-2 w-full accent-brand-600"
                />
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
