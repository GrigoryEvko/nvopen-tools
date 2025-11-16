// Function: sub_1317140
// Address: 0x1317140
//
char *__fastcall sub_1317140(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        _QWORD *a4,
        __int64 *a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15)
{
  __int64 v15; // rcx
  _QWORD *v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // r10
  __int64 v22; // rdx
  _QWORD *v23; // r14
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // r11
  __int64 v27; // r10
  __int64 v28; // r9
  __int64 v29; // rcx
  _WORD *v30; // rsi
  __int16 v31; // di
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rdx
  __int64 v34; // r14
  const __m128i *v35; // rax
  __int64 v36; // rax
  __int64 v37; // r13
  unsigned int v38; // ebx
  unsigned int v39; // eax
  __int64 v40; // r14
  char *result; // rax
  pthread_mutex_t *mutex; // [rsp+28h] [rbp-70h]
  int v45; // [rsp+38h] [rbp-60h]
  char *v46; // [rsp+38h] [rbp-60h]
  __int64 v47; // [rsp+48h] [rbp-50h] BYREF
  __int64 v48; // [rsp+50h] [rbp-48h] BYREF
  __int64 v49; // [rsp+58h] [rbp-40h] BYREF
  _QWORD v50[7]; // [rsp+60h] [rbp-38h] BYREF

  sub_1317090(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  sub_131C490(a1, *(_QWORD *)(a2 + 78936), &v47, &v48, &v49, v50);
  v15 = a2 + 976;
  *(_QWORD *)(a10 + 24) += *(_QWORD *)(*(_QWORD *)(a2 + 72896) + 56LL) + v49;
  *(_QWORD *)(a10 + 8) += v48;
  *(_QWORD *)a10 += v47;
  *(_QWORD *)(a10 + 32) += *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a10 + 16) += v50[0];
  v16 = &qword_505FA40[36];
  v17 = (_QWORD *)(a12 + 8);
  do
  {
    v18 = *(_QWORD *)(v15 + 8);
    ++v16;
    v15 += 48;
    *v17 += v18;
    *(_QWORD *)(a10 + 56) += v18;
    v19 = *(_QWORD *)(v15 - 48);
    *(v17 - 1) += v19;
    *(_QWORD *)(a10 + 48) += v19;
    v20 = v19 + *(_QWORD *)(v15 - 32);
    v17[1] += v20;
    *(_QWORD *)(a10 + 80) += v20;
    v17[2] += v19;
    *(_QWORD *)(a10 + 64) += v19;
    v21 = *(_QWORD *)(v15 - 16);
    v22 = v19 - v18;
    v17[3] += v21;
    *(_QWORD *)(a10 + 72) += v21;
    v17[4] += v22;
    v17 += 6;
    *(_QWORD *)(a10 + 40) += *(v16 - 1) * v22;
  }
  while ( v16 != qword_5060180 );
  sub_134AE90(a1, a2 + 10648, a10 + 88, a13, a14, a15, a10 + 8);
  *(_QWORD *)(a10 + 168) = 0;
  *(_QWORD *)(a10 + 176) = 0;
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 10472)) )
  {
    sub_130AD90(a2 + 10408);
    *(_BYTE *)(a2 + 10512) = 1;
  }
  ++*(_QWORD *)(a2 + 10464);
  if ( a1 != *(_QWORD *)(a2 + 10456) )
  {
    ++*(_QWORD *)(a2 + 10448);
    *(_QWORD *)(a2 + 10456) = a1;
  }
  v23 = *(_QWORD **)(a2 + 10400);
  v24 = dword_5060A18[0];
  v45 = dword_5060A18[0];
  v25 = unk_5060A20;
  do
  {
    if ( !v23 )
      break;
    if ( v45 )
    {
      v26 = *(_QWORD *)(a10 + 168);
      v27 = *(_QWORD *)(a10 + 176);
      v28 = 0;
      v29 = 0;
      do
      {
        v30 = (_WORD *)(v28 + v23[2]);
        v28 += 24;
        v31 = v30[10];
        v26 += qword_505FA40[v29] * ((unsigned __int16)(v31 - *v30) >> 3);
        v32 = (unsigned __int64)((unsigned __int16)v30[9]
                               - (unsigned int)(unsigned __int16)(v31 - 8 * *(_WORD *)(v25 + 2 * v29))) << 45 >> 48;
        *(_QWORD *)(a10 + 168) = v26;
        v33 = qword_505FA40[v29++] * v32;
        v27 += v33;
        *(_QWORD *)(a10 + 176) = v27;
      }
      while ( v24 != v29 );
    }
    v23 = (_QWORD *)*v23;
  }
  while ( v23 != *(_QWORD **)(a2 + 10400) );
  *(__m128i *)(a10 + 696) = _mm_loadu_si128((const __m128i *)(a2 + 10408));
  *(__m128i *)(a10 + 712) = _mm_loadu_si128((const __m128i *)(a2 + 10424));
  *(__m128i *)(a10 + 728) = _mm_loadu_si128((const __m128i *)(a2 + 10440));
  *(__m128i *)(a10 + 744) = _mm_loadu_si128((const __m128i *)(a2 + 10456));
  *(_DWORD *)(a10 + 732) = 0;
  *(_BYTE *)(a2 + 10512) = 0;
  pthread_mutex_unlock((pthread_mutex_t *)(a2 + 10472));
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 10600)) )
  {
    sub_130AD90(a2 + 10536);
    *(_BYTE *)(a2 + 10640) = 1;
  }
  ++*(_QWORD *)(a2 + 10592);
  if ( a1 != *(_QWORD *)(a2 + 10584) )
  {
    ++*(_QWORD *)(a2 + 10576);
    *(_QWORD *)(a2 + 10584) = a1;
  }
  *(__m128i *)(a10 + 184) = _mm_loadu_si128((const __m128i *)(a2 + 10536));
  *(__m128i *)(a10 + 200) = _mm_loadu_si128((const __m128i *)(a2 + 10552));
  *(__m128i *)(a10 + 216) = _mm_loadu_si128((const __m128i *)(a2 + 10568));
  *(__m128i *)(a10 + 232) = _mm_loadu_si128((const __m128i *)(a2 + 10584));
  *(_DWORD *)(a10 + 220) = 0;
  *(_BYTE *)(a2 + 10640) = 0;
  pthread_mutex_unlock((pthread_mutex_t *)(a2 + 10600));
  v34 = *(_QWORD *)(a2 + 78936);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(v34 + 96)) )
  {
    sub_130AD90(v34 + 32);
    *(_BYTE *)(v34 + 136) = 1;
  }
  ++*(_QWORD *)(v34 + 88);
  if ( a1 != *(_QWORD *)(v34 + 80) )
  {
    ++*(_QWORD *)(v34 + 72);
    *(_QWORD *)(v34 + 80) = a1;
  }
  v35 = *(const __m128i **)(a2 + 78936);
  *(__m128i *)(a10 + 632) = _mm_loadu_si128(v35 + 2);
  *(__m128i *)(a10 + 648) = _mm_loadu_si128(v35 + 3);
  *(__m128i *)(a10 + 664) = _mm_loadu_si128(v35 + 4);
  *(__m128i *)(a10 + 680) = _mm_loadu_si128(v35 + 5);
  *(_DWORD *)(a10 + 668) = 0;
  v36 = *(_QWORD *)(a2 + 78936);
  *(_BYTE *)(v36 + 136) = 0;
  pthread_mutex_unlock((pthread_mutex_t *)(v36 + 96));
  sub_134B160(a1, a2 + 10648, a10 + 184);
  sub_130B140((__int64 *)(a10 + 10360), (__int64 *)(a2 + 78944));
  sub_130B160((__int64 *)(a10 + 10360));
  sub_130B1F0((_QWORD *)(a10 + 10360), (__int64 *)(a2 + 78944));
  v46 = (char *)&unk_5260DF4;
  v37 = a11 + 116;
  mutex = (pthread_mutex_t *)dword_5060A40;
  do
  {
    v38 = 0;
    while ( *(_DWORD *)v46 > v38 )
    {
      v40 = (unsigned int)mutex->__lock + a2 + 224LL * v38;
      if ( pthread_mutex_trylock((pthread_mutex_t *)(v40 + 64)) )
      {
        sub_130AD90(v40);
        *(_BYTE *)(v40 + 104) = 1;
      }
      ++*(_QWORD *)(v40 + 56);
      if ( a1 != *(_QWORD *)(v40 + 48) )
      {
        ++*(_QWORD *)(v40 + 40);
        *(_QWORD *)(v40 + 48) = a1;
      }
      sub_130B1D0((_QWORD *)(v37 - 36), (__int64 *)v40);
      if ( (int)sub_130B150((_QWORD *)(v40 + 8), (_QWORD *)(v37 - 28)) > 0 )
        sub_130B140((__int64 *)(v37 - 28), (__int64 *)(v40 + 8));
      *(_QWORD *)(v37 - 20) += *(_QWORD *)(v40 + 16);
      *(_QWORD *)(v37 - 12) += *(_QWORD *)(v40 + 24);
      v39 = *(_DWORD *)(v40 + 32);
      if ( *(_DWORD *)(v37 - 4) < v39 )
        *(_DWORD *)(v37 - 4) = v39;
      *(_DWORD *)v37 = 0;
      ++v38;
      *(_QWORD *)(v37 + 4) += *(_QWORD *)(v40 + 40);
      *(_QWORD *)(v37 + 20) += *(_QWORD *)(v40 + 56);
      *(_QWORD *)(v37 - 116) += *(_QWORD *)(v40 + 112);
      *(_QWORD *)(v37 - 108) += *(_QWORD *)(v40 + 120);
      *(_QWORD *)(v37 - 100) += *(_QWORD *)(v40 + 128);
      *(_QWORD *)(v37 - 92) += *(_QWORD *)(v40 + 136);
      *(_QWORD *)(v37 - 84) += *(_QWORD *)(v40 + 144);
      *(_QWORD *)(v37 - 76) += *(_QWORD *)(v40 + 152);
      *(_QWORD *)(v37 - 68) += *(_QWORD *)(v40 + 160);
      *(_QWORD *)(v37 - 60) += *(_QWORD *)(v40 + 168);
      *(_QWORD *)(v37 - 52) += *(_QWORD *)(v40 + 176);
      *(_QWORD *)(v37 - 44) += *(_QWORD *)(v40 + 184);
      *(_BYTE *)(v40 + 104) = 0;
      pthread_mutex_unlock((pthread_mutex_t *)(v40 + 64));
    }
    v46 += 40;
    v37 += 144;
    result = v46;
    mutex = (pthread_mutex_t *)((char *)mutex + 4);
  }
  while ( (char *)&unk_5260DE0 + 1460 != v46 );
  return result;
}
