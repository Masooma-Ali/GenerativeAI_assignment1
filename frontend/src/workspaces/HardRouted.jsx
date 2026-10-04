import { useState } from 'react'
import { BRANCH_LABELS, CLASS_LABELS } from '../api'
import RestorationWorkspace from '../components/RestorationWorkspace'
import { Card, KeyValue, Label, Segmented, WeightBars } from '../components/ui'

const PIPELINE = ['clean', 'salt_pepper', 'blur', 'occlusion']

export default function HardRouted() {
  const [mode, setMode] = useState('predicted')

  return (
    <RestorationWorkspace
      endpoint="/api/restore/hard"
      downloadPrefix="hard_routed"
      extraFields={{ mode }}
      controls={
        <div>
          <Label>Routing mode</Label>
          <Segmented
            value={mode}
            onChange={setMode}
            options={[
              { value: 'predicted', label: 'Predicted (classifier)' },
              { value: 'oracle', label: 'Oracle (true label)' },
            ]}
          />
          <p className="mt-1.5 text-xs text-slate-500">
            Oracle routing uses the known corruption label and needs a corruption applied in the app.
          </p>
        </div>
      }
      insights={(r) => (
        <Card title="Classifier & routing" subtitle="p = C(x̃), r = argmax p" className="lg:col-span-2">
          <div className="grid gap-6 md:grid-cols-[1fr_260px]">
            <WeightBars values={r.probabilities} labels={CLASS_LABELS} trueLabel={r.true_label} />
            <div>
              <KeyValue
                rows={[
                  ['Mode', r.mode],
                  ['Predicted', CLASS_LABELS[r.predicted]],
                  ['Confidence', `${(r.confidence * 100).toFixed(2)}%`],
                  ['True label', r.true_label ? CLASS_LABELS[r.true_label] : 'unknown'],
                  ['Selected expert', r.selected_expert],
                ]}
              />
              {r.misrouted && (
                <div className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
                  ⚠ Classifier prediction differs from the true corruption
                  {r.mode === 'oracle' ? ' (oracle mode overrode it).' : ' — this is a routing error.'}
                </div>
              )}
            </div>
          </div>
          <div className="mt-6">
            <Label>Route</Label>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-lg bg-slate-100 px-3 py-2 font-medium">Input</span>
              <span className="text-slate-400">→</span>
              <span className="rounded-lg bg-slate-100 px-3 py-2 font-medium">Classifier</span>
              <span className="text-slate-400">→</span>
              <div className="flex flex-wrap gap-2">
                {PIPELINE.map((k) => (
                  <span
                    key={k}
                    className={`rounded-lg px-3 py-2 font-medium ${
                      r.routed_to === k ? 'bg-brand-600 text-white shadow' : 'bg-white text-slate-400 ring-1 ring-slate-200'
                    }`}
                  >
                    {BRANCH_LABELS[k]}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}
    />
  )
}
