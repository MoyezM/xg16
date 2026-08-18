//! Streaming chunker: feed data in arbitrary pieces, receive chunks as
//! they complete. Produces byte-identical chunks to slice chunking — a
//! cut decision needs at most `max_size` bytes of lookahead, so a chunk
//! is emitted as soon as that much data is buffered past its start.

use crate::chunker::Chunker;

/// Incremental chunker over a byte stream.
///
/// ```
/// use xg16::{Chunker, StreamChunker};
///
/// let data = vec![7u8; 100_000];
/// let mut lens = Vec::new();
/// let mut s = StreamChunker::new(Chunker::with_default_sizes());
/// for piece in data.chunks(4096) {
///     s.push(piece, |chunk| lens.push(chunk.len()));
/// }
/// s.finish(|chunk| lens.push(chunk.len()));
/// assert_eq!(lens.iter().sum::<usize>(), data.len());
/// ```
pub struct StreamChunker {
    chunker: Chunker,
    buf: Vec<u8>,
    start: usize,
}

impl StreamChunker {
    pub fn new(chunker: Chunker) -> Self {
        StreamChunker {
            chunker,
            buf: Vec::new(),
            start: 0,
        }
    }

    /// Feed more input. `emit` is called once per completed chunk, in
    /// order. Buffered memory stays bounded by roughly
    /// `2 * (max_size + largest push)`.
    pub fn push(&mut self, data: &[u8], mut emit: impl FnMut(&[u8])) {
        self.buf.extend_from_slice(data);
        // With at least max_size bytes available, the next cut cannot
        // depend on unseen input (scans never read past max_size).
        while self.buf.len() - self.start >= self.chunker.max_size() {
            let cut = self.chunker.next_cut(&self.buf[self.start..]);
            emit(&self.buf[self.start..self.start + cut]);
            self.start += cut;
        }
        if self.start > 0 && self.start >= self.buf.len() / 2 {
            self.buf.drain(..self.start);
            self.start = 0;
        }
    }

    /// End of input: emit the remaining chunks (the final one is the
    /// EOF-forced chunk and may be shorter than `min_size`).
    pub fn finish(self, mut emit: impl FnMut(&[u8])) {
        let mut rest = &self.buf[self.start..];
        while !rest.is_empty() {
            let cut = self.chunker.next_cut(rest);
            let (c, r) = rest.split_at(cut);
            emit(c);
            rest = r;
        }
    }
}
