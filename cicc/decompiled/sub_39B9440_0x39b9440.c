// Function: sub_39B9440
// Address: 0x39b9440
//
__int64 *__fastcall sub_39B9440(__int64 *a1, __int64 a2)
{
  const __m128i *v4; // r14
  const __m128i *v5; // rbx
  int v6; // eax
  unsigned int v7; // ecx
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  const __m128i *v12; // rcx
  __int64 v13; // rsi
  const __m128i *v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  _BYTE *v17; // rdi
  __m128i *v18; // rsi
  const __m128i *v19; // rax
  __int64 v20; // rsi
  _QWORD *v21; // rdi
  unsigned int v22; // eax
  int v23; // eax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  int v26; // ebx
  unsigned __int64 v27; // r14
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *j; // rdx
  _QWORD *v31; // rax

  v4 = *(const __m128i **)(a2 + 8);
  v5 = &v4[*(unsigned int *)(a2 + 24)];
  if ( !*(_DWORD *)(a2 + 16) || v4 == v5 )
    goto LABEL_2;
  while ( v4->m128i_i64[0] == -8 || v4->m128i_i64[0] == -16 )
  {
    if ( v5 == ++v4 )
      goto LABEL_2;
  }
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v5 == v4 )
  {
LABEL_2:
    *a1 = 0;
    a1[2] = 0;
    a1[1] = 0;
  }
  else
  {
    v12 = v4;
    v13 = 0;
    while ( 1 )
    {
      v14 = v12 + 1;
      if ( v5 == &v12[1] )
        break;
      while ( 1 )
      {
        v12 = v14;
        if ( v14->m128i_i64[0] != -8 && v14->m128i_i64[0] != -16 )
          break;
        if ( v5 == ++v14 )
          goto LABEL_25;
      }
      ++v13;
      if ( v5 == v14 )
        goto LABEL_26;
    }
LABEL_25:
    ++v13;
LABEL_26:
    if ( v13 > 0x7FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v15 = 16 * v13;
    v16 = sub_22077B0(v15);
    *a1 = v16;
    v17 = (_BYTE *)v16;
    a1[2] = v16 + v15;
    v18 = (__m128i *)v16;
    while ( 1 )
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v4);
      v19 = v4 + 1;
      if ( v5 == &v4[1] )
        break;
      while ( 1 )
      {
        v4 = v19;
        if ( v19->m128i_i64[0] != -16 && v19->m128i_i64[0] != -8 )
          break;
        if ( v5 == ++v19 )
          goto LABEL_34;
      }
      ++v18;
      if ( v5 == v19 )
        goto LABEL_35;
    }
LABEL_34:
    ++v18;
LABEL_35:
    a1[1] = (__int64)v18;
    v20 = (char *)v18 - v17;
    if ( v20 > 16 )
    {
      qsort(v17, v20 >> 4, 0x10u, (__compar_fn_t)sub_39B9370);
      v6 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      if ( !v6 )
        goto LABEL_11;
      goto LABEL_4;
    }
  }
  v6 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  if ( !v6 )
  {
LABEL_11:
    if ( !*(_DWORD *)(a2 + 20) )
      return a1;
    v8 = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)v8 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a2 + 8));
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      *(_DWORD *)(a2 + 24) = 0;
      return a1;
    }
    goto LABEL_7;
  }
LABEL_4:
  v7 = 4 * v6;
  v8 = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v7 = 64;
  if ( v7 >= (unsigned int)v8 )
  {
LABEL_7:
    v9 = *(_QWORD **)(a2 + 8);
    for ( i = &v9[2 * v8]; i != v9; v9 += 2 )
      *v9 = -8;
    *(_QWORD *)(a2 + 16) = 0;
    return a1;
  }
  v21 = *(_QWORD **)(a2 + 8);
  v22 = v6 - 1;
  if ( !v22 )
  {
    v27 = 2048;
    v26 = 128;
LABEL_47:
    j___libc_free_0((unsigned __int64)v21);
    *(_DWORD *)(a2 + 24) = v26;
    v28 = (_QWORD *)sub_22077B0(v27);
    v29 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 8) = v28;
    for ( j = &v28[2 * v29]; j != v28; v28 += 2 )
    {
      if ( v28 )
        *v28 = -8;
    }
    return a1;
  }
  _BitScanReverse(&v22, v22);
  v23 = 1 << (33 - (v22 ^ 0x1F));
  if ( v23 < 64 )
    v23 = 64;
  if ( (_DWORD)v8 != v23 )
  {
    v24 = (4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1);
    v25 = ((v24 | (v24 >> 2)) >> 4) | v24 | (v24 >> 2) | ((((v24 | (v24 >> 2)) >> 4) | v24 | (v24 >> 2)) >> 8);
    v26 = (v25 | (v25 >> 16)) + 1;
    v27 = 16 * ((v25 | (v25 >> 16)) + 1);
    goto LABEL_47;
  }
  *(_QWORD *)(a2 + 16) = 0;
  v31 = &v21[2 * (unsigned int)v8];
  do
  {
    if ( v21 )
      *v21 = -8;
    v21 += 2;
  }
  while ( v31 != v21 );
  return a1;
}
