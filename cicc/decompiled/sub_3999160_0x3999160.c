// Function: sub_3999160
// Address: 0x3999160
//
__int64 __fastcall sub_3999160(__int64 a1, const __m128i *a2)
{
  __int64 v4; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 *v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  int v13; // eax
  int v14; // edx
  __m128i *v15; // rsi
  __m128i *v16; // rsi
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rsi
  int v22; // r9d
  __int64 *v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // r8d
  unsigned int v28; // r15d
  __int64 *v29; // rdi
  __int64 v30; // rcx

  v4 = a2->m128i_i64[0];
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
    return *(_QWORD *)(a1 + 32) + 16LL * *((unsigned int *)v10 + 2);
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      return *(_QWORD *)(a1 + 32) + 16LL * *((unsigned int *)v10 + 2);
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_23:
    sub_154E4B0(a1, 2 * v5);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 8);
      v20 = (v17 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v19 + 16LL * v20);
      v21 = *v8;
      if ( v4 != *v8 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -8 )
        {
          if ( !v23 && v21 == -16 )
            v23 = v8;
          v20 = v18 & (v22 + v20);
          v8 = (__int64 *)(v19 + 16LL * v20);
          v21 = *v8;
          if ( v4 == *v8 )
            goto LABEL_14;
          ++v22;
        }
        if ( v23 )
          v8 = v23;
      }
      goto LABEL_14;
    }
    goto LABEL_46;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v14 <= v5 >> 3 )
  {
    sub_154E4B0(a1, v5);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v27 = 1;
      v28 = v25 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v29 = 0;
      v8 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v8;
      if ( v4 != *v8 )
      {
        while ( v30 != -8 )
        {
          if ( !v29 && v30 == -16 )
            v29 = v8;
          v28 = v25 & (v27 + v28);
          v8 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v8;
          if ( v4 == *v8 )
            goto LABEL_14;
          ++v27;
        }
        if ( v29 )
          v8 = v29;
      }
      goto LABEL_14;
    }
LABEL_46:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  *((_DWORD *)v8 + 2) = 0;
  v15 = *(__m128i **)(a1 + 40);
  if ( v15 == *(__m128i **)(a1 + 48) )
  {
    sub_398F2D0((unsigned __int64 *)(a1 + 32), v15, a2);
    v16 = *(__m128i **)(a1 + 40);
  }
  else
  {
    if ( v15 )
    {
      *v15 = _mm_loadu_si128(a2);
      v15 = *(__m128i **)(a1 + 40);
    }
    v16 = v15 + 1;
    *(_QWORD *)(a1 + 40) = v16;
  }
  *((_DWORD *)v8 + 2) = (((__int64)v16->m128i_i64 - *(_QWORD *)(a1 + 32)) >> 4) - 1;
  return *(_QWORD *)(a1 + 40) - 16LL;
}
