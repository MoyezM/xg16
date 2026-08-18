//! Streaming chunking over any [`std::io::Read`], following the
//! `fastcdc-rs` `StreamCDC` shape: an iterator of `Result<ChunkData>`.
//!
//! Cuts are byte-identical to slice chunking: a cut decision needs at
//! most `max_size` bytes of lookahead, so a chunk is emitted as soon as
//! that much data is buffered past its start (or the source ends).
//! Buffered memory stays bounded by roughly `2 * max_size`.

use std::io::Read;

use crate::chunker::Config;

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
/// `Result<ChunkData, Error>`.
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
    config: Config,
    source: R,
    buf: Vec<u8>,
    start: usize,
    offset: u64,
    eof: bool,
    failed: bool,
}

impl<R: Read> StreamChunker<R> {
    /// Create a streaming chunker.
    ///
    /// # Panics
    ///
    /// Same size constraints as [`crate::Xg16::new`].
    pub fn new(source: R, min_size: usize, avg_size: usize, max_size: usize) -> Self {
        StreamChunker {
            config: Config::new(min_size, avg_size, max_size),
            source,
            buf: Vec::new(),
            start: 0,
            offset: 0,
            eof: false,
            failed: false,
        }
    }

    /// Top up the buffer until a cut is decidable or the source ends.
    fn fill(&mut self) -> Result<(), Error> {
        let mut tmp = [0u8; 32 * 1024];
        while !self.eof && self.buf.len() - self.start < self.config.max_size {
            match self.source.read(&mut tmp) {
                Ok(0) => self.eof = true,
                Ok(n) => self.buf.extend_from_slice(&tmp[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
}

impl<R: Read> Iterator for StreamChunker<R> {
    type Item = Result<ChunkData, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        if let Err(e) = self.fill() {
            self.failed = true;
            return Some(Err(e));
        }
        let avail = self.buf.len() - self.start;
        if avail == 0 {
            return None;
        }
        // Either max_size bytes are buffered (cut is lookahead-complete)
        // or the source ended (EOF semantics apply).
        let (length, hash) = self.config.cut(&self.buf[self.start..]);
        let chunk = ChunkData {
            hash,
            offset: self.offset,
            length,
            data: self.buf[self.start..self.start + length].to_vec(),
        };
        self.start += length;
        self.offset += length as u64;
        if self.start >= self.buf.len() / 2 {
            self.buf.drain(..self.start);
            self.start = 0;
        }
        Some(Ok(chunk))
    }
}
