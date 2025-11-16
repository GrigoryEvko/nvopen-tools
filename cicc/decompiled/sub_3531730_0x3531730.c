// Function: sub_3531730
// Address: 0x3531730
//
__int64 *__fastcall sub_3531730(__int64 *a1, __int64 a2)
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
  unsigned int v21; // eax
  _QWORD *v22; // rdi
  int v23; // ebx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *j; // rdx
  _QWORD *v29; // rax

  v4 = *(const __m128i **)(a2 + 8);
  v5 = &v4[*(unsigned int *)(a2 + 24)];
  if ( !*(_DWORD *)(a2 + 16) || v4 == v5 )
    goto LABEL_2;
  while ( v4->m128i_i64[0] == -4096 || v4->m128i_i64[0] == -8192 )
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
        if ( v14->m128i_i64[0] != -4096 && v14->m128i_i64[0] != -8192 )
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
        if ( v19->m128i_i64[0] != -8192 && v19->m128i_i64[0] != -4096 )
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
      qsort(v17, v20 >> 4, 0x10u, (__compar_fn_t)sub_35316B0);
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
      sub_C7D6A0(*(_QWORD *)(a2 + 8), 16LL * *(unsigned int *)(a2 + 24), 8);
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
      *v9 = -4096;
    *(_QWORD *)(a2 + 16) = 0;
    return a1;
  }
  v21 = v6 - 1;
  if ( !v21 )
  {
    v22 = *(_QWORD **)(a2 + 8);
    v23 = 64;
LABEL_46:
    sub_C7D6A0((__int64)v22, 16LL * *(unsigned int *)(a2 + 24), 8);
    v24 = ((((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 16;
    v25 = (v24
         | (((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a2 + 24) = v25;
    v26 = (_QWORD *)sub_C7D670(16 * v25, 8);
    v27 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 8) = v26;
    for ( j = &v26[2 * v27]; j != v26; v26 += 2 )
    {
      if ( v26 )
        *v26 = -4096;
    }
    return a1;
  }
  _BitScanReverse(&v21, v21);
  v22 = *(_QWORD **)(a2 + 8);
  v23 = 1 << (33 - (v21 ^ 0x1F));
  if ( v23 < 64 )
    v23 = 64;
  if ( v23 != (_DWORD)v8 )
    goto LABEL_46;
  *(_QWORD *)(a2 + 16) = 0;
  v29 = &v22[2 * (unsigned int)v23];
  do
  {
    if ( v22 )
      *v22 = -4096;
    v22 += 2;
  }
  while ( v29 != v22 );
  return a1;
}
