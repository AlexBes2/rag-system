import { useEffect, useMemo, useState } from 'react'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Привет! Задай вопрос по загруженным документам.',
      sources: [],
    },
  ])
  const [loading, setLoading] = useState(false)
  const [documents, setDocuments] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('')

  const canSend = useMemo(() => {
    return question.trim().length > 0 && !loading
  }, [question, loading])

  useEffect(() => {
    fetchDocuments()
  }, [])

  async function fetchDocuments() {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`)
      const data = await response.json()

      const docs = Array.isArray(data)
        ? data
        : Array.isArray(data.documents)
        ? data.documents
        : []

      setDocuments(docs)
    } catch (error) {
      console.error('Ошибка загрузки списка документов:', error)
    }
  }

  function getDocumentLabel(doc, index) {
    if (typeof doc === 'string') {
      return doc
    }

    return doc.filename || doc.document_name || doc.name || `Документ ${index + 1}`
  }

  function getSourceLabel(source, index) {
    if (typeof source === 'string') {
      return source
    }

    const documentName =
      source.document_name || source.filename || source.document || source.source

    const page =
      source.page !== undefined && source.page !== null
        ? `стр. ${source.page}`
        : null

    const parts = [documentName, page].filter(Boolean)

    return parts.length > 0 ? parts.join(' · ') : `Источник ${index + 1}`
  }

  function getSourceUrl(source) {
    if (typeof source === 'string') {
      return null
    }

    if (source.url) return source.url
    if (source.file_url) return source.file_url

    const documentName =
      source.document_name || source.filename || source.document || source.source

    if (documentName) {
      return `${API_BASE_URL}/files/${encodeURIComponent(documentName)}`
    }

    return null
  }

  function dedupeSources(sources) {
    if (!Array.isArray(sources)) {
      return []
    }

    const unique = []
    const seen = new Set()

    for (const source of sources) {
      if (typeof source === 'string') {
        const key = `string:${source}`
        if (seen.has(key)) continue
        seen.add(key)
        unique.push(source)
        continue
      }

      const documentName =
        source.document_name ||
        source.filename ||
        source.document ||
        source.source ||
        'unknown'

      const page =
        source.page !== undefined && source.page !== null
          ? String(source.page)
          : 'no-page'

      const key = `${documentName}::${page}`

      if (seen.has(key)) {
        continue
      }

      seen.add(key)
      unique.push(source)
    }

    return unique
  }

  async function handleSubmit(event) {
    event.preventDefault()

    const userQuestion = question.trim()
    if (!userQuestion || loading) return

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userQuestion, sources: [] },
    ])
    setQuestion('')
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userQuestion,
          k: 3,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Ошибка запроса')
      }

      const data = await response.json()

      const answer =
        data.answer ||
        data.response ||
        data.text ||
        data.result ||
        'Сервер вернул пустой ответ.'

      const rawSources = Array.isArray(data.sources) ? data.sources : []
      const sources = dedupeSources(rawSources)

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: answer, sources },
      ])
    } catch (error) {
      console.error(error)

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Ошибка: ${error.message || 'не удалось получить ответ от сервера'}`,
          sources: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  async function handleUpload(event) {
    event.preventDefault()

    if (!selectedFile) {
      setUploadStatus('Выбери файл перед загрузкой.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    setUploadStatus('Загрузка...')

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Ошибка загрузки файла')
      }

      const data = await response.json()
      setUploadStatus(data.message || 'Файл успешно загружен.')
      setSelectedFile(null)
      await fetchDocuments()
    } catch (error) {
      console.error(error)
      setUploadStatus(`Ошибка: ${error.message || 'не удалось загрузить файл'}`)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="panel">
          <h2>Документы</h2>

          <form onSubmit={handleUpload} className="upload-form">
            <input
              type="file"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
            />
            <button type="submit">Загрузить</button>
          </form>

          {uploadStatus && <p className="status">{uploadStatus}</p>}

          <div className="documents-list">
            {documents.length === 0 ? (
              <p className="muted">Пока нет документов.</p>
            ) : (
              <ul>
                {documents.map((doc, index) => {
                  const label = getDocumentLabel(doc, index)
                  return <li key={`${label}-${index}`}>{label}</li>
                })}
              </ul>
            )}
          </div>
        </div>
      </aside>

      <main className="chat-area">
        <div className="chat-header">
          <h1>RAG Chat</h1>
          <p>Вопросы по загруженным документам</p>
        </div>

        <div className="messages">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role === 'user' ? 'user' : 'assistant'}`}
            >
              <div className="message-role">
                {message.role === 'user' ? 'Ты' : 'AI'}
              </div>

              <div className="message-content">{message.content}</div>

              {message.role === 'assistant' && message.sources?.length > 0 && (
                <div className="sources">
                  <div className="sources-title">Источники:</div>
                  <ul>
                    {message.sources.map((source, sourceIndex) => {
                      const label = getSourceLabel(source, sourceIndex)
                      const url = getSourceUrl(source)

                      return (
                        <li key={`${label}-${sourceIndex}`}>
                          {url ? (
                            <a href={url} target="_blank" rel="noreferrer">
                              {label}
                            </a>
                          ) : (
                            <span>{label}</span>
                          )}
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-role">AI</div>
              <div className="message-content">Думаю над ответом...</div>
            </div>
          )}
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Введите вопрос..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button type="submit" disabled={!canSend}>
            {loading ? '...' : 'Отправить'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default App