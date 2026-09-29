/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import sympy_primes from './primes';

// Simplified RSA for demonstration
export interface RSAKey {
  n: bigint;
  e: bigint;
  p: bigint;
  q: bigint;
  d: bigint;
}

function gcd(a: bigint, b: bigint): bigint {
  while (b > 0n) {
    a %= b;
    [a, b] = [b, a];
  }
  return a;
}

function modInverse(e: bigint, phi: bigint): bigint {
  let m0 = phi;
  let y = 0n, x = 1n;
  if (phi === 1n) return 0n;
  while (e > 1n) {
    let q = e / phi;
    let t = phi;
    phi = e % phi;
    e = t;
    t = y;
    y = x - q * y;
    x = t;
  }
  if (x < 0n) x += m0;
  return x;
}

export function generateKeys(bits: number = 8): RSAKey {
  if (bits !== 8) throw new Error('Only the 8-bit-prime demonstration is implemented');

  // Primes from 2^(bits-1) to 2^bits
  const min = Math.pow(2, bits - 1);
  const max = Math.pow(2, bits);
  
  const selectablePrimes = sympy_primes.filter(p => p >= min && p <= max);
  
  const p = BigInt(selectablePrimes[Math.floor(Math.random() * selectablePrimes.length)]);
  let q = BigInt(selectablePrimes[Math.floor(Math.random() * selectablePrimes.length)]);
  while (p === q) {
    q = BigInt(selectablePrimes[Math.floor(Math.random() * selectablePrimes.length)]);
  }
  
  const n = p * q;
  const phi = (p - 1n) * (q - 1n);
  
  let e = 65537n;
  if (e >= phi) {
    e = 3n;
    while (gcd(e, phi) !== 1n) e += 2n;
  }
  
  const d = modInverse(e, phi);
  
  return { n, e, p, q, d };
}

// Energy function for visualization
export function getEnergy(x: number, y: number, n: number): number {
  const base = Math.pow((n - x * y) / Math.sqrt(n), 2);
  const int_pen = 10.0 * (Math.pow(Math.sin(Math.PI * x), 2) + Math.pow(Math.sin(Math.PI * y), 2));
  return base + int_pen;
}

// Simulated solver step
export interface SolverStep {
  x: number;
  y: number;
  type: 'gradient' | 'mdn_jump' | 'success';
  energy: number;
  message: string;
}
