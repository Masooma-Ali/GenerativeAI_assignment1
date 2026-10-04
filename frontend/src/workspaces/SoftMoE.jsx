import { BRANCH_LABELS } from '../api'
import RestorationWorkspace from '../components/RestorationWorkspace'
import { Card, KeyValue, WeightBars } from '../components/ui'

const COLORS = { clean: '#10b981', salt_pepper: '#f59e0b', blur: '#0ea5e9', occlusion: '#d946ef' }

export default function SoftMoE() {
  return (
    <RestorationWorkspace
      endpoint="/api/restore/soft"
      downloadPrefix="soft_moe"
      outputTitle="Mixture"
      insights={(r) => (
        <Card
          title="Routing weights"
          subtitle="w = softmax(G(x̃)/τ);  x̂ = w₀x̃ + w₁A_salt + w₂A_blur + w₃A_occ"
          className="lg:col-span-2"
        >
          {/* stacked contribution strip */}
          <div className="mb-5 flex h-8 overflow-hidden rounded-lg ring-1 ring-slate-200">
            {r.ranking.map((k) => (
              <div
                key={k}
                title={`${BRANCH_LABELS[k]}: ${(r.weights[k] * 100).toFixed(1)}%`}
                className="flex items-center justify-center text-[11px] font-semibold text-white transition-all duration-500"
                style={{ width: `${r.weights[k] * 100}%`, background: COLORS[k] }}
              >
                {r.weights[k] > 0.08 ? `${(r.weights[k] * 100).toFixed(0)}%` : ''}
              </div>
            ))}
          </div>
          <div className="grid gap-6 md:grid-cols-[1fr_260px]">
            <WeightBars values={r.weights} labels={BRANCH_LABELS} highlight={r.dominant} trueLabel={r.true_label} />
            <KeyValue
              rows={[
                ['Dominant branch', r.dominant_label],
                ['Ranking', r.ranking.map((k) => BRANCH_LABELS[k].replace(' expert', '')).join(' > ')],
                ['Routing entropy', `${r.routing_entropy} (0 = one-hot, 1 = uniform)`],
                ['True corruption', r.true_label ?? 'unknown'],
                ['Routing', r.routing_entropy < 0.3 ? 'Sharp — one expert dominates' : 'Distributed across experts'],
              ]}
            />
          </div>
        </Card>
      )}
    />
  )
}
