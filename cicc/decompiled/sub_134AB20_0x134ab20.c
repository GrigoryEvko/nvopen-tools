// Function: sub_134AB20
// Address: 0x134ab20
//
int __fastcall sub_134AB20(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  pthread_mutex_t *v4; // r15
  __int64 v5; // r13
  __m128i *v7; // rsi

  v4 = (pthread_mutex_t *)(a3 + 64);
  v5 = a4;
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a3 + 64)) )
  {
    sub_130AD90(a3);
    *(_BYTE *)(a3 + 104) = 1;
  }
  ++*(_QWORD *)(a3 + 56);
  if ( a1 != *(_QWORD *)(a3 + 48) )
  {
    ++*(_QWORD *)(a3 + 40);
    *(_QWORD *)(a3 + 48) = a1;
  }
  v7 = (__m128i *)(a2 + (v5 << 6));
  *v7 = _mm_loadu_si128((const __m128i *)a3);
  v7[1] = _mm_loadu_si128((const __m128i *)(a3 + 16));
  v7[2] = _mm_loadu_si128((const __m128i *)(a3 + 32));
  v7[3] = _mm_loadu_si128((const __m128i *)(a3 + 48));
  v7[2].m128i_i32[1] = 0;
  *(_BYTE *)(a3 + 104) = 0;
  return pthread_mutex_unlock(v4);
}
