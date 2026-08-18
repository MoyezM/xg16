//! Streaming chunking, in two layers:
//!
//! - [`Feeder`]: push-based (`push(&bytes)* , finish()`), for producers
//!   that deliver bytes in irregular pieces from mixed or asynchronous
//!   sources. Chunks are emitted with stream-relative offsets alongside
//!   their bytes.
//! - [`StreamChunker`]: wraps any [`std::io::Read`] and yields owned
//!   [`ChunkData`]; implemented on top of [`Feeder`].
//!
//! Both layers produce chunks byte-identical to slice chunking: a cut
//! decision needs at most `max_size` bytes of lookahead, so a chunk is
//! emitted as soon as that much data is buffered past its start (or the
//! input ends). Buffered memory stays bounded by roughly
//! `2 * (max_size + largest push)`.

use std::io::Read;

use crate::chunker::{Chunk, Config};

/// Push-based chunker. Feed bytes in pieces of any size; completed
/// chunks are handed to `emit` with stream-relative offsets and their
/// bytes (still hot — hashing them in the callback is the intended
/// pattern; see the crate docs on why there is no fused-hash hook).
///
/// ```
/// use xg16::Feeder;
///
/// let data = vec![7u8; 100_000];
/// let mut total = 0;
/// let mut f = Feeder::new(2048, 8192, 65536);
/// for piece in data.chunks(4096) {
///     f.push(piece, |chunk, _bytes| total += chunk.length);
/// }
/// f.finish(|chunk, _bytes| total += chunk.length);
/// assert_eq!(total, data.len());
/// ```
pub struct Feeder {
    config: Config,
    buf: Vec<u8>,
    start: usize,
    offset: usize,
}

impl Feeder {
    /// # Panics
    ///
    /// Same size constraints as [`crate::Xg16::new`].
    pub fn new(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        Feeder {
            config: Config::new(min_size, avg_size, max_size),
            buf: Vec::new(),
            start: 0,
            offset: 0,
        }
    }

    /// The configured minimum chunk size.
    pub fn min_size(&self) -> usize {
        self.config.min_size
    }

    /// The configured target (average) chunk size.
    pub fn avg_size(&self) -> usize {
        self.config.avg_size
    }

    /// The configured maximum chunk size.
    pub fn max_size(&self) -> usize {
        self.config.max_size
    }

    /// Feed more input; `emit` is called once per completed chunk, in
    /// stream order.
    pub fn push(&mut self, data: &[u8], mut emit: impl FnMut(Chunk, &[u8])) {
        self.buf.extend_from_slice(data);
        // With max_size bytes buffered, the next cut cannot depend on
        // unseen input (scans never read past max_size).
        while self.buf.len() - self.start >= self.config.max_size {
            let (length, hash) = self.config.cut(&self.buf[self.start..]);
            let chunk = Chunk {
                hash,
                offset: self.offset,
                length,
            };
            emit(chunk, &self.buf[self.start..self.start + length]);
            self.start += length;
            self.offset += length;
        }
        if self.start > 0 && self.start >= self.buf.len() / 2 {
            self.buf.drain(..self.start);
            self.start = 0;
        }
    }

    /// End of input: emit the remaining chunks (the final one is the
    /// EOF-forced chunk and may be shorter than `min_size`).
    pub fn finish(mut self, mut emit: impl FnMut(Chunk, &[u8])) {
        while self.start < self.buf.len() {
            let (length, hash) = self.config.cut(&self.buf[self.start..]);
            let chunk = Chunk {
                hash,
                offset: self.offset,
                length,
            };
            emit(chunk, &self.buf[self.start..self.start + length]);
            self.start += length;
            self.offset += length;
        }
    }
}

/// An owned chunk produced from a stream. Field semantics match
/// [`crate::Chunk`]; `data` holds the chunk bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkData {
    /// Rolling-hash state at the cut (boundary artifact, not a content
    /// fingerprint — see [`crate::Chunk`]).
    pub hash: u64,
    /// Byte offset of the chunk from the start of the stream.
    pub offset: u64,
    /// Length of the chunk in bytes.
    pub length: usize,
    /// The chunk bytes.
    pub data: Vec<u8>,
}

/// Streaming error.
#[derive(Debug)]
pub enum Error {
    /// Reading from the source failed.
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "read error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Content-defined chunker over a byte stream; iterates over
/// `Result<ChunkData, Error>`. A convenience layer over [`Feeder`] for
/// blocking `Read` sources.
///
/// ```
/// use std::io::Cursor;
/// use xg16::StreamChunker;
///
/// let data = vec![7u8; 100_000];
/// let mut total = 0;
/// for chunk in StreamChunker::new(Cursor::new(&data), 2048, 8192, 65536) {
///     total += chunk.unwrap().length;
/// }
/// assert_eq!(total, data.len());
/// ```
pub struct StreamChunker<R: Read> {
    feeder: Option<Feeder>,
    source: R,
    ready: std::collections::VecDeque<ChunkData>,
    failed: bool,
}

impl<R: Read> StreamChunker<R> {
    /// # Panics
    ///
    /// Same size constraints as [`crate::Xg16::new`].
    pub fn new(source: R, min_size: usize, avg_size: usize, max_size: usize) -> Self {
        StreamChunker {
            feeder: Some(Feeder::new(min_size, avg_size, max_size)),
            source,
            ready: std::collections::VecDeque::new(),
            failed: false,
        }
    }
}

fn to_chunk_data(chunk: Chunk, bytes: &[u8]) -> ChunkData {
    ChunkData {
        hash: chunk.hash,
        offset: chunk.offset as u64,
        length: chunk.length,
        data: bytes.to_vec(),
    }
}

impl<R: Read> Iterator for StreamChunker<R> {
    type Item = Result<ChunkData, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(c) = self.ready.pop_front() {
                return Some(Ok(c));
            }
            if self.failed {
                return None;
            }
            let feeder = self.feeder.as_mut()?;
            let mut tmp = [0u8; 32 * 1024];
            match self.source.read(&mut tmp) {
                Ok(0) => {
                    let feeder = self.feeder.take().expect("checked above");
                    let ready = &mut self.ready;
                    feeder.finish(|c, bytes| ready.push_back(to_chunk_data(c, bytes)));
                }
                Ok(n) => {
                    let ready = &mut self.ready;
                    feeder.push(&tmp[..n], |c, bytes| {
                        ready.push_back(to_chunk_data(c, bytes))
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.failed = true;
                    return Some(Err(e.into()));
                }
            }
        }
    }
}
