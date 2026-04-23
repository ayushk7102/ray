use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_pyarrow::PyArrowType;
use arrow_schema::{ArrowError, Schema, SchemaRef};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::fs::File;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// RecordBatchReader that drains a sequence of per-row-group bounded channels
/// in order. Decode happens on background worker threads, each holding its
/// own synchronous `ParquetRecordBatchReader` for a single row group.
///
/// Because the sync reader is page-streamed, per-worker memory stays ~12 MiB
/// regardless of row group size; total in-flight memory is bounded by
/// `num_threads * per_worker_footprint + num_row_groups * channel_capacity * batch_size`.
struct SyncParquetReader {
    schema: SchemaRef,
    receivers: VecDeque<Receiver<Result<RecordBatch, ArrowError>>>, // bounded channels for each row group
    current_receiver: Option<Receiver<Result<RecordBatch, ArrowError>>>, // current receiver
    /// Shared with worker threads. Cleared on drop so detached workers see an
    /// empty queue and exit their loop instead of decoding remaining row groups
    /// into already-dropped receivers.
    work_queue: Arc<Mutex<VecDeque<(usize, usize)>>>, // work queue of (output_position, row_group_index) which is the unit of work
    _workers: Vec<JoinHandle<()>>,
}

impl Drop for SyncParquetReader {
    fn drop(&mut self) {
        if let Ok(mut q) = self.work_queue.lock() {
            q.clear();
        }
    }
}

impl Iterator for SyncParquetReader {
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // if there is no current receiver, grab the first one from the queue
            if self.current_receiver.is_none() {
                self.current_receiver = self.receivers.pop_front();
                if self.current_receiver.is_none() {
                    return None;
                }
            }
            // receive the batch from the current receiver
            match self.current_receiver.as_ref().unwrap().recv() {
                Ok(batch) => return Some(batch),
                Err(_) => self.current_receiver = None,
            }
        }
    }
}

impl RecordBatchReader for SyncParquetReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[pyfunction]
fn read_parquet_schema(path: &str) -> PyResult<PyArrowType<Schema>> {
    let file = File::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let meta = ArrowReaderMetadata::load(&file, ArrowReaderOptions::default())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrowType(meta.schema().as_ref().clone()))
}

/// Decode `row_group_idx` from `path` and drain batches into `sender`.
///
/// Accepts pre-loaded `ArrowReaderMetadata` so the parquet footer is not
/// re-read on every call (the footer is shared via `Arc` inside the metadata).
fn decode_row_group(
    path: &str,
    row_group_idx: usize,
    batch_size: usize,
    projection: Option<ProjectionMask>,
    meta: ArrowReaderMetadata,
    sender: SyncSender<Result<RecordBatch, ArrowError>>,
) {
    let build = || -> Result<_, ArrowError> {
        let file = File::open(path)?;
        let mut builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, meta)
            .with_row_groups(vec![row_group_idx])
            .with_batch_size(batch_size);
        if let Some(mask) = projection {
            builder = builder.with_projection(mask);
        }
        Ok(builder.build()?)
    };
    match build() {
        Err(e) => {
            let _ = sender.send(Err(e));
        }
        Ok(reader) => {
            for batch in reader {
                if sender.send(batch).is_err() {
                    return;
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (path, columns=None, row_groups=None, batch_size=65536, num_threads=4))]
fn read_parquet_batches_stream(
    path: &str,
    columns: Option<Vec<String>>,
    row_groups: Option<Vec<usize>>,
    batch_size: usize,
    num_threads: usize,
) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
    let file = File::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?;

    // Read the parquet footer exactly once. ArrowReaderMetadata::load takes
    // &file so `file` remains owned and available for the projection probe below.
    // Workers clone this metadata (cheap: Arc<ParquetMetaData> + SchemaRef) so
    // they skip the footer re-read entirely.
    let arrow_meta = ArrowReaderMetadata::load(&file, ArrowReaderOptions::default())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let rg_indices: Vec<usize> = match row_groups {
        Some(rgs) => rgs,
        None => (0..arrow_meta.metadata().num_row_groups()).collect(),
    };

    let projection: Option<ProjectionMask> = if let Some(cols) = columns {
        let pq_schema = arrow_meta.metadata().file_metadata().schema_descr();
        for name in &cols {
            let name_path: Vec<&str> = name.split('.').collect();
            let matched = pq_schema.columns().iter().any(|col| {
                let path = col.path().parts();
                name_path.len() <= path.len()
                    && name_path.iter().zip(path.iter()).all(|(a, b)| a == b)
            });
            if !matched {
                return Err(PyValueError::new_err(format!(
                    "column {name:?} not found in parquet schema"
                )));
            }
        }
        Some(ProjectionMask::columns(
            pq_schema,
            cols.iter().map(|s| s.as_str()),
        ))
    } else {
        None
    };

    // Derive the schema that batches will carry. Without a projection the full
    // schema comes directly from `arrow_meta` (no I/O). With a projection we
    // build a zero-work probe reader from the already-open `file` to let the
    // builder compute the projected schema; the file is closed when the probe
    // is dropped.
    let schema: SchemaRef = if let Some(ref mask) = projection {
        let probe = ParquetRecordBatchReaderBuilder::new_with_metadata(file, arrow_meta.clone())
            .with_projection(mask.clone())
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        probe.schema()
    } else {
        arrow_meta.schema().clone()
    };

    // Per-row-group bounded channels; capacity bounds the decoded-batch backlog
    // ahead of the consumer for each row group. With CAP=1, the total rust-side
    // in-flight memory is `num_threads * batch_size`, since a worker blocks on
    // send once its channel holds one batch until the consumer drains it.
    const PER_RG_CHANNEL_CAP: usize = 1;
    let (senders, receivers): (Vec<_>, VecDeque<_>) = (0..rg_indices.len())
        .map(|_| mpsc::sync_channel::<Result<RecordBatch, ArrowError>>(PER_RG_CHANNEL_CAP))
        .unzip();
    let senders: Arc<Mutex<Vec<Option<SyncSender<Result<RecordBatch, ArrowError>>>>>> =
        Arc::new(Mutex::new(senders.into_iter().map(Some).collect()));

    // FIFO work queue of (output_pos, row_group_idx). FIFO ordering ensures
    // workers process the row groups the consumer will want next.
    let work_queue: Arc<Mutex<VecDeque<(usize, usize)>>> =
        Arc::new(Mutex::new(rg_indices.into_iter().enumerate().collect()));

    let num_threads = num_threads.max(1);
    let path_arc: Arc<str> = Arc::from(path);
    let mut workers = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let work_queue = work_queue.clone();
        let senders = senders.clone();
        let path = path_arc.clone();
        let projection = projection.clone();
        let meta = arrow_meta.clone();
        let worker = thread::spawn(move || {
            loop {
                // Claim the next (output slot, row group) pair from the FIFO.
                // An empty queue means no more work — exit the worker thread.
                let Some((output_pos, row_group_idx)) = work_queue.lock().unwrap().pop_front()
                else {
                    return;
                };
                // Take ownership of the matching sender. `None` only happens if
                // the slot was already claimed by another worker; skip and move on.
                let Some(sender) = senders.lock().unwrap()[output_pos].take() else {
                    continue;
                };
                decode_row_group(
                    &path,
                    row_group_idx,
                    batch_size,
                    projection.clone(),
                    meta.clone(),
                    sender,
                );
            }
        });
        workers.push(worker);
    }

    Ok(PyArrowType(Box::new(SyncParquetReader {
        schema,
        receivers,
        current_receiver: None,
        work_queue,
        _workers: workers,
    }) as Box<dyn RecordBatchReader + Send>))
}

#[pymodule]
fn ray_parquet_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_parquet_schema, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet_batches_stream, m)?)?;
    Ok(())
}
