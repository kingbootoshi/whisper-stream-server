/**
 * Batch transcription endpoint tests
 *
 * Sends a full WAV file and validates JSON response shape + timestamps.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import { getFixturePath } from '../utils/WavLoader.js';

interface BatchWordTiming {
  word: string;
  start_ms: number;
  end_ms: number;
  probability: number;
}

interface BatchTranscribeResponse {
  text: string;
  duration_ms: number;
  processing_ms: number;
  words: BatchWordTiming[];
}

const WS_URL = process.env.WHISPER_SERVER_URL ?? 'ws://localhost:9090';
const HTTP_URL = process.env.WHISPER_HTTP_URL ?? WS_URL;
const TOKEN = process.env.WHISPER_SERVER_TOKEN ?? '';

function toHttpUrl(base: string): string {
  const url = new URL(base);
  if (url.protocol === 'ws:') url.protocol = 'http:';
  if (url.protocol === 'wss:') url.protocol = 'https:';
  url.pathname = '/v1/transcribe';
  url.search = '';
  url.hash = '';
  return url.toString();
}

describe.sequential('Batch Transcribe', () => {
  it('returns text + word timestamps for JFK WAV', async () => {
    const wavPath = getFixturePath('jfk.wav');
    const wav = fs.readFileSync(wavPath);

    const headers: Record<string, string> = {
      'Content-Type': 'audio/wav',
    };
    if (TOKEN) {
      headers.Authorization = `Bearer ${TOKEN}`;
    }

    const res = await fetch(toHttpUrl(HTTP_URL), {
      method: 'POST',
      headers,
      body: wav,
    });

    if (res.status === 401 && !TOKEN) {
      console.warn('[batch-transcribe] Skipping: WHISPER_SERVER_TOKEN not set');
      return;
    }

    console.log('[batch-transcribe] status', res.status);
    const data = (await res.json()) as BatchTranscribeResponse;
    console.log('[batch-transcribe] response', data);

    expect(res.status).toBe(200);

    expect(typeof data.text).toBe('string');
    expect(data.text.length).toBeGreaterThan(0);
    expect(data.duration_ms).toBeGreaterThan(1000);
    expect(data.processing_ms).toBeGreaterThanOrEqual(0);

    const lower = data.text.toLowerCase();
    expect(lower.includes('ask') || lower.includes('country')).toBe(true);

    expect(Array.isArray(data.words)).toBe(true);
    expect(data.words.length).toBeGreaterThan(0);

    let lastStart = 0;
    for (const w of data.words) {
      expect(w.word.length).toBeGreaterThan(0);
      expect(w.word.includes('[_')).toBe(false);
      expect(w.word.includes('<|')).toBe(false);
      expect(w.start_ms).toBeGreaterThanOrEqual(0);
      expect(w.end_ms).toBeGreaterThanOrEqual(w.start_ms);
      expect(w.probability).toBeGreaterThan(0);
      expect(w.probability).toBeLessThanOrEqual(1);
      expect(w.start_ms).toBeGreaterThanOrEqual(lastStart);
      lastStart = w.start_ms;
    }
  });
});
