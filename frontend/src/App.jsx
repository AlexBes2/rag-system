import { useEffect, useMemo, useState } from 'react'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const DEMO_QUESTIONS = [
  'Who are the authors of the paper "LEPA: Learning Geometric Equivariance in Satellite Remote Sensing Data with a Predictive Architecture"?',
  'Summarize the paper "Deep Expert Injection for Anchoring Retinal VLMs with Domain-Specific Knowledge" in 2-3 sentences.',
  'What problem does the paper "Experiences Build Characters: The Linguistic Origins and Functional Impact of LLM Personality" try to solve?',
  'What method or system is proposed in "Progressive Residual Warmup for Language Model Pretraining"?',
  'Summarize the paper "Benchmarking Large Language Models for Quebec Insurance: From Closed-Book to Retrieval-Augmented Generation" in 2-3 sentences.',
]

const SUPPORTED_UPLOAD_EXTENSIONS = [
  '.pdf',
  '.docx',
  '.png',
  '.jpg',
  '.jpeg',
  '.tif',
  '.tiff',
  '.bmp',
]
const SUPPORTED_FILE_ACCEPT = SUPPORTED_UPLOAD_EXTENSIONS.join(',')
const UPLOAD_BATCH_SIZE = 10

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
  const [documentsMeta, setDocumentsMeta] = useState({
    totalDocuments: 0,
    totalChunks: 0,
  })
  const [selectedFiles, setSelectedFiles] = useState([])
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState({
    total: 0,
    processed: 0,
    success: 0,
    failed: 0,
  })
  const [pickerKey, setPickerKey] = useState(0)

  const canSend = useMemo(() => {
    return question.trim().length > 0 && !loading
  }, [question, loading])

  const canUpload = useMemo(() => {
    return selectedFiles.length > 0 && !uploading
  }, [selectedFiles, uploading])

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
      setDocumentsMeta({
        totalDocuments:
          typeof data.total_documents === 'number'
            ? data.total_documents
            : docs.length,
        totalChunks:
          typeof data.total_chunks === 'number' ? data.total_chunks : 0,
      })
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

  function getDocumentMeta(doc) {
    if (!doc || typeof doc === 'string') {
      return null
    }

    const parts = []
    if (typeof doc.chunk_count === 'number') {
      parts.push(`${doc.chunk_count} чанков`)
    }
    if (Array.isArray(doc.pages) && doc.pages.length > 0) {
      parts.push(`${doc.pages.length} стр.`)
    }

    return parts.length > 0 ? parts.join(' · ') : null
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
    const chunk =
      source.chunk_id !== undefined && source.chunk_id !== null
        ? `chunk ${source.chunk_id}`
        : null

    const parts = [documentName, page, chunk].filter(Boolean)

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

  function isSupportedUploadFile(file) {
    const name = file?.name || ''
    const lowerName = name.toLowerCase()
    return SUPPORTED_UPLOAD_EXTENSIONS.some((extension) =>
      lowerName.endsWith(extension),
    )
  }

  function getFileKey(file) {
    return [
      file.webkitRelativePath || file.name,
      file.size,
      file.lastModified,
    ].join(':')
  }

  function handleSelectedFiles(fileList) {
    const incomingFiles = Array.from(fileList || [])
    const selected = []
    const seen = new Set()
    let skipped = 0

    for (const file of incomingFiles) {
      if (!isSupportedUploadFile(file)) {
        skipped += 1
        continue
      }

      const key = getFileKey(file)
      if (seen.has(key)) {
        continue
      }

      seen.add(key)
      selected.push(file)
    }

    setSelectedFiles(selected)
    setUploadProgress({
      total: selected.length,
      processed: 0,
      success: 0,
      failed: 0,
    })

    if (selected.length === 0) {
      setUploadStatus(
        skipped > 0
          ? `Поддерживаемых файлов не найдено. Пропущено: ${skipped}.`
          : 'Файлы не выбраны.',
      )
      return
    }

    setUploadStatus(
      `Выбрано файлов: ${selected.length}${
        skipped > 0 ? `. Пропущено неподдерживаемых: ${skipped}.` : '.'
      }`,
    )
  }

  function chunkFiles(files, size) {
    const chunks = []

    for (let index = 0; index < files.length; index += size) {
      chunks.push(files.slice(index, index + size))
    }

    return chunks
  }

  async function uploadFileBatch(files) {
    const formData = new FormData()

    for (const file of files) {
      formData.append('files', file, file.webkitRelativePath || file.name)
    }

    const response = await fetch(`${API_BASE_URL}/upload/batch`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || 'Ошибка загрузки файлов')
    }

    return response.json()
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

    if (selectedFiles.length === 0) {
      setUploadStatus('Выбери файлы или папку перед загрузкой.')
      return
    }

    const total = selectedFiles.length
    const batches = chunkFiles(selectedFiles, UPLOAD_BATCH_SIZE)
    let success = 0
    let failed = 0
    let processed = 0
    let chunksIndexed = 0

    setUploading(true)
    setUploadStatus(`Загрузка: 0 из ${total}`)
    setUploadProgress({ total, processed: 0, success: 0, failed: 0 })

    try {
      for (const batch of batches) {
        const data = await uploadFileBatch(batch)

        const batchSuccess =
          typeof data.success_count === 'number' ? data.success_count : 0
        const batchFailed =
          typeof data.failed_count === 'number' ? data.failed_count : 0

        success += batchSuccess
        failed += batchFailed
        processed += batch.length
        chunksIndexed +=
          typeof data.chunks_indexed === 'number' ? data.chunks_indexed : 0

        setUploadProgress({ total, processed, success, failed })
        setUploadStatus(
          `Загрузка: ${processed} из ${total}. Успешно: ${success}, ошибок: ${failed}.`,
        )
      }

      setSelectedFiles([])
      setPickerKey((current) => current + 1)
      await fetchDocuments()
      setUploadStatus(
        `Готово: загружено ${success} из ${total}, ошибок: ${failed}, чанков: ${chunksIndexed}.`,
      )
    } catch (error) {
      console.error(error)
      setUploadStatus(`Ошибка: ${error.message || 'не удалось загрузить файлы'}`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="panel">
          <h2>Документы</h2>
          <div className="documents-summary">
            <strong>{documentsMeta.totalDocuments}</strong>
            <span>документов</span>
            {documentsMeta.totalChunks > 0 && (
              <>
                <strong>{documentsMeta.totalChunks}</strong>
                <span>чанков</span>
              </>
            )}
          </div>

          <form onSubmit={handleUpload} className="upload-form">
            <div className="upload-pickers">
              <label className="file-picker">
                Файлы
                <input
                  key={`files-${pickerKey}`}
                  type="file"
                  multiple
                  accept={SUPPORTED_FILE_ACCEPT}
                  onChange={(e) => handleSelectedFiles(e.target.files)}
                />
              </label>

              <label className="file-picker">
                Папка
                <input
                  key={`folder-${pickerKey}`}
                  type="file"
                  multiple
                  webkitdirectory=""
                  directory=""
                  onChange={(e) => handleSelectedFiles(e.target.files)}
                />
              </label>
            </div>

            {selectedFiles.length > 0 && (
              <div className="selected-files">
                <strong>{selectedFiles.length}</strong>
                <span>к загрузке</span>
              </div>
            )}

            {uploadProgress.total > 0 && (
              <div className="upload-progress">
                <div className="progress-track">
                  <div
                    className="progress-bar"
                    style={{
                      width: `${Math.round(
                        (uploadProgress.processed / uploadProgress.total) * 100,
                      )}%`,
                    }}
                  />
                </div>
                <span>
                  {uploadProgress.processed}/{uploadProgress.total}
                </span>
              </div>
            )}

            <button type="submit" disabled={!canUpload}>
              {uploading ? 'Загружаю...' : 'Загрузить'}
            </button>
          </form>

          {uploadStatus && <p className="status">{uploadStatus}</p>}

          <div className="documents-list">
            {documents.length === 0 ? (
              <p className="muted">Пока нет документов.</p>
            ) : (
              <ul>
                {documents.map((doc, index) => {
                  const label = getDocumentLabel(doc, index)
                  const meta = getDocumentMeta(doc)
                  return (
                    <li key={`${label}-${index}`}>
                      <span>{label}</span>
                      {meta && <small>{meta}</small>}
                    </li>
                  )
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
          <div className="demo-questions">
            {DEMO_QUESTIONS.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setQuestion(item)}
                disabled={loading}
              >
                {item}
              </button>
            ))}
          </div>
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
                          {source.snippet && (
                            <p className="source-snippet">{source.snippet}</p>
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
