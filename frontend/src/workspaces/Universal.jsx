import RestorationWorkspace from '../components/RestorationWorkspace'

export default function Universal() {
  return <RestorationWorkspace endpoint="/api/restore/universal" downloadPrefix="universal" />
}
