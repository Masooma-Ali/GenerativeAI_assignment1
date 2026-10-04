const BASE = import.meta.env.VITE_API_BASE || ''

async function handle(res) {
  let body = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON error */
  }
  if (!res.ok) {
    const detail = body?.detail
    const msg = typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `${res.status} ${res.statusText}`
    throw new Error(msg)
  }
  return body
}

export async function getJSON(path) {
  return handle(await fetch(BASE + path))
}

export async function postForm(path, fields) {
  const fd = new FormData()
  for (const [k, v] of Object.entries(fields)) {
    if (v !== undefined && v !== null && v !== '') fd.append(k, v)
  }
  return handle(await fetch(BASE + path, { method: 'POST', body: fd }))
}

export async function fetchSampleFile(sample) {
  const res = await fetch(BASE + sample.url)
  if (!res.ok) throw new Error(`Could not load sample ${sample.name}`)
  const blob = await res.blob()
  return new File([blob], sample.name, { type: blob.type || 'image/png' })
}

export const CLASS_LABELS = {
  clean: 'Clean',
  salt_pepper: 'Salt & pepper',
  blur: 'Gaussian blur',
  occlusion: 'Occlusion',
}

export const BRANCH_LABELS = {
  clean: 'Identity',
  salt_pepper: 'Salt & pepper expert',
  blur: 'Blur expert',
  occlusion: 'Occlusion expert',
}
