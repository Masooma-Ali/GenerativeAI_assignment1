import { useEffect, useRef, useState } from 'react'
import { fetchSampleFile, getJSON } from '../api'
import { Button } from './ui'

/**
 * Upload / drag-and-drop / sample picker / optional webcam capture.
 * Calls onChange(File | null). Shows a preview of the chosen file.
 */
export default function ImageInput({ file, onChange, allowWebcam = false, allowSamples = true }) {
  const [preview, setPreview] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [samples, setSamples] = useState([])
  const [camOpen, setCamOpen] = useState(false)
  const [camError, setCamError] = useState(null)
  const inputRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(() => {
    if (!file) return setPreview(null)
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!allowSamples) return
    getJSON('/api/samples')
      .then((r) => setSamples(r.samples || []))
      .catch(() => setSamples([]))
  }, [allowSamples])

  useEffect(() => () => stopCam(), [])

  function pick(f) {
    if (f && f.type.startsWith('image/')) onChange(f)
  }

  async function startCam() {
    setCamError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      streamRef.current = stream
      setCamOpen(true)
      requestAnimationFrame(() => {
        if (videoRef.current) videoRef.current.srcObject = stream
      })
    } catch (e) {
      setCamError(`Webcam unavailable: ${e.message}`)
    }
  }

  function stopCam() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setCamOpen(false)
  }

  function capture() {
    const v = videoRef.current
    if (!v) return
    // centre square crop so the face is not stretched when the backend resizes to 128x128
    const s = Math.min(v.videoWidth, v.videoHeight)
    const c = document.createElement('canvas')
    c.width = c.height = s
    const ctx = c.getContext('2d')
    ctx.translate(s, 0)
    ctx.scale(-1, 1) // un-mirror the selfie view
    ctx.drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, s, s)
    c.toBlob((b) => {
      onChange(new File([b], `webcam_${Date.now()}.png`, { type: 'image/png' }))
      stopCam()
    }, 'image/png')
  }

  return (
    <div className="space-y-3">
      {camOpen ? (
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-black">
          <video ref={videoRef} autoPlay playsInline muted className="aspect-video w-full -scale-x-100 object-cover" />
          <div className="flex justify-center gap-2 bg-slate-900 p-2">
            <Button onClick={capture}>Capture</Button>
            <Button variant="secondary" onClick={stopCam}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            pick(e.dataTransfer.files?.[0])
          }}
          className={`group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-4 text-center transition ${
            dragging ? 'border-brand-500 bg-brand-50' : 'border-slate-300 bg-slate-50 hover:border-brand-500 hover:bg-brand-50/40'
          }`}
        >
          {preview ? (
            <>
              <img src={preview} alt="selected" className="max-h-48 rounded-lg object-contain" />
              <p className="mt-2 max-w-full truncate text-xs text-slate-500">
                {file.name} · {(file.size / 1024).toFixed(0)} KB · click to change
              </p>
            </>
          ) : (
            <div className="py-6">
              <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-white text-xl text-brand-600 shadow-sm">↑</div>
              <p className="text-sm font-medium text-slate-700">Drop an image or click to upload</p>
              <p className="mt-1 text-xs text-slate-500">PNG, JPG, BMP, WEBP · max 10 MB · resized to 128×128</p>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              pick(e.target.files?.[0])
              e.target.value = ''
            }}
          />
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {allowWebcam && !camOpen && (
          <Button variant="secondary" onClick={startCam} className="text-xs">
            ● Use webcam
          </Button>
        )}
        {file && (
          <Button variant="ghost" onClick={() => onChange(null)} className="text-xs">
            Clear
          </Button>
        )}
      </div>
      {camError && <p className="text-xs text-red-600">{camError}</p>}

      {allowSamples && samples.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Or pick a clean sample</p>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {samples.map((s) => (
              <button
                key={s.name}
                title={s.name}
                onClick={async () => onChange(await fetchSampleFile(s))}
                className={`h-14 w-14 shrink-0 overflow-hidden rounded-lg border-2 transition ${
                  file?.name === s.name ? 'border-brand-500' : 'border-transparent hover:border-slate-300'
                }`}
              >
                <img src={s.url} alt={s.name} className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
