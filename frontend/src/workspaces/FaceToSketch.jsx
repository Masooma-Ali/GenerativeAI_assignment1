import { useState } from 'react'
import { postForm } from '../api'
import ImageInput from '../components/ImageInput'
import { Button, Card, ErrorBanner, ImageCard, KeyValue, Label, Segmented, Spinner, Stat } from '../components/ui'

export default function FaceToSketch() {
  const [file, setFile] = useState(null)
  const [style, setStyle] = useState('1')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function run() {
    setLoading(true)
    setError(null)
    try {
      setResult(await postForm('/api/sketch', { file, style }))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[360px_1fr]">
      <div className="space-y-6">
        <Card title="1 · Facial photograph" subtitle="Upload a photo or capture one with your webcam">
          <ImageInput file={file} onChange={setFile} allowWebcam allowSamples={false} />
        </Card>
        <Card title="2 · Sketch style">
          <Label>Style condition</Label>
          <Segmented
            value={style}
            onChange={setStyle}
            options={[
              { value: '1', label: 'Style 1' },
              { value: '2', label: 'Style 2' },
              { value: '3', label: 'Style 3' },
              { value: 'all', label: 'Compare all' },
            ]}
          />
          <p className="mt-2 text-xs text-slate-500">
            The style is a learned categorical embedding fed to the U-Net generator (FS2K style categories).
          </p>
          <Button className="mt-5 w-full" onClick={run} disabled={!file || loading}>
            {loading ? <Spinner /> : '✎'} {loading ? 'Generating…' : 'Generate sketch'}
          </Button>
        </Card>
      </div>

      <div className="space-y-6">
        <ErrorBanner error={error} />
        {!result && !error && (
          <div className="flex h-full min-h-[300px] items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 text-sm text-slate-400">
            Generated sketch will appear here
          </div>
        )}
        {result && (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              <Stat label="Inference" value={result.timing.inference_ms} unit="ms" tone="brand" />
              <Stat label="Total request" value={result.timing.total_ms} unit="ms" />
              <Stat label="Sketches" value={result.sketches.length} />
            </div>
            <Card title="Photo → Sketch">
              <div className={`grid gap-4 sm:grid-cols-2 ${result.sketches.length > 1 ? 'lg:grid-cols-4' : 'lg:grid-cols-2'}`}>
                <ImageCard title="Photo (128×128)" src={result.input_image} />
                {result.sketches.map((s) => (
                  <ImageCard
                    key={s.style}
                    title={s.name}
                    src={s.image}
                    downloadName={`sketch_style${s.style}.png`}
                    highlight
                  />
                ))}
              </div>
            </Card>
            <Card title="System information">
              <KeyValue
                rows={[
                  ['ONNX model', result.model],
                  ['Original size', result.original_size.join(' × ')],
                  ['Processed size', result.processed_size.join(' × ')],
                  ['Preprocess', `${result.timing.preprocess_ms} ms`],
                  ['Inference', `${result.timing.inference_ms} ms`],
                  ['Total (server)', `${result.timing.total_ms} ms`],
                ]}
              />
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
