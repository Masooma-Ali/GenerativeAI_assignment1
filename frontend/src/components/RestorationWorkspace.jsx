import { useState } from 'react'
import { postForm } from '../api'
import CorruptionControls, { DEFAULT_CORRUPTION, corruptionFields } from './CorruptionControls'
import ImageInput from './ImageInput'
import { Button, Card, ErrorBanner, ImageCard, KeyValue, Spinner, Stat } from './ui'

function fmt(v, d = 2) {
  return v === null || v === undefined ? '∞' : Number(v).toFixed(d)
}

function corruptionRows(c) {
  if (!c) return [['Corruption', 'None applied (upload used as-is)']]
  const rows = [
    ['Type', c.type],
    ['Severity', c.severity],
    ['Seed', c.seed],
  ]
  if (c.p !== undefined) rows.push(['Probability p', c.p])
  if (c.kernel !== undefined) rows.push(['Kernel / σ', `${c.kernel}×${c.kernel} / ${c.sigma}`])
  if (c.boxes) {
    rows.push(['Rectangles', c.n_rects])
    rows.push(['Coverage', `${(c.coverage * 100).toFixed(1)}%`])
    rows.push(['Boxes [y0,x0,y1,x1]', <span className="font-mono text-xs">{c.boxes.map((b) => `[${b.join(',')}]`).join(' ')}</span>])
  }
  return rows
}

/**
 * Shared layout for Tasks 1-3: input + corruption controls on the left, results on the right.
 * `extraFields` are appended to the request, `controls` renders extra inputs, and
 * `insights(result)` renders the task-specific panel (probabilities, MoE weights, ...).
 */
export default function RestorationWorkspace({
  endpoint,
  downloadPrefix,
  extraFields = {},
  controls,
  insights,
  outputTitle = 'Restored output',
}) {
  const [file, setFile] = useState(null)
  const [corr, setCorr] = useState(DEFAULT_CORRUPTION)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function run() {
    setLoading(true)
    setError(null)
    try {
      setResult(await postForm(endpoint, { file, ...corruptionFields(corr), ...extraFields }))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const m = result?.metrics
  return (
    <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
      <div className="space-y-6">
        <Card title="1 · Input image" subtitle="Upload a corrupted image, or a clean one and corrupt it below">
          <ImageInput file={file} onChange={setFile} />
        </Card>
        <Card title="2 · Corruption settings">
          <CorruptionControls value={corr} onChange={setCorr} />
          {controls && <div className="mt-4 border-t border-slate-100 pt-4">{controls}</div>}
          <Button className="mt-5 w-full" onClick={run} disabled={!file || loading}>
            {loading ? <Spinner /> : '▶'} {loading ? 'Running…' : 'Run model'}
          </Button>
        </Card>
      </div>

      <div className="space-y-6">
        <ErrorBanner error={error} />
        {!result && !error && (
          <div className="flex h-full min-h-[300px] items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 text-sm text-slate-400">
            Results will appear here
          </div>
        )}
        {result && (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label="Inference" value={result.timing.inference_ms} unit="ms" tone="brand" />
              <Stat label="Total request" value={result.timing.total_ms} unit="ms" />
              {m ? (
                <>
                  <Stat
                    label="PSNR in → out"
                    value={`${fmt(m.input.psnr, 1)} → ${fmt(m.output.psnr, 1)}`}
                    unit="dB"
                    tone={m.output.psnr === null || (m.input.psnr !== null && m.output.psnr >= m.input.psnr) ? 'green' : 'red'}
                  />
                  <Stat
                    label="SSIM in → out"
                    value={`${fmt(m.input.ssim, 3)} → ${fmt(m.output.ssim, 3)}`}
                    tone={m.output.ssim >= m.input.ssim ? 'green' : 'red'}
                  />
                </>
              ) : (
                <div className="col-span-2 flex items-center rounded-xl border border-slate-200 bg-white px-4 text-xs text-slate-500">
                  No clean target → quality metrics unavailable
                </div>
              )}
            </div>

            <Card title="Images" subtitle="All panels are the 128×128 model resolution">
              <div className={`grid gap-4 sm:grid-cols-2 ${result.clean_image ? 'lg:grid-cols-4' : 'lg:grid-cols-3'}`}>
                {result.clean_image && <ImageCard title="Clean target" src={result.clean_image} />}
                <ImageCard title="Model input" src={result.input_image} downloadName={`${downloadPrefix}_input.png`} />
                <ImageCard title={outputTitle} src={result.output_image} downloadName={`${downloadPrefix}_restored.png`} highlight />
                <ImageCard
                  title="Absolute error map"
                  src={result.error_map}
                  caption={`|${result.error_map_reference === 'clean target' ? 'clean' : 'input'} − output|, ×3 gain`}
                />
              </div>
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              {insights && insights(result)}
              <Card title="Corruption settings">
                <KeyValue rows={corruptionRows(result.corruption)} />
              </Card>
              <Card title="System information">
                <KeyValue
                  rows={[
                    ['ONNX model', result.model || result.expert_model || '—'],
                    ['Original size', result.original_size.join(' × ')],
                    ['Processed size', result.processed_size.join(' × ')],
                    ['Preprocess', `${result.timing.preprocess_ms} ms`],
                    ...(result.timing.classifier_ms !== undefined
                      ? [
                          ['Classifier', `${result.timing.classifier_ms} ms`],
                          ['Expert', `${result.timing.expert_ms} ms`],
                        ]
                      : []),
                    ['Inference', `${result.timing.inference_ms} ms`],
                    ['Total (server)', `${result.timing.total_ms} ms`],
                  ]}
                />
              </Card>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
