// Function: sub_32A7D60
// Address: 0x32a7d60
//
bool __fastcall sub_32A7D60(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __int64 v9; // rax
  _DWORD *v10; // rcx
  __int64 v11; // rax
  int v12; // ecx
  __m128i v13; // xmm0
  __int64 v14; // rax
  __m128i v15; // xmm0
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // esi
  int v19; // ecx
  __m128i v20; // xmm0
  __int64 v21; // rsi
  __m128i v22; // xmm0
  __int64 v23; // rsi
  __int64 v24; // rax
  int v25; // esi
  __int64 v26; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v4 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v5 = *(_QWORD *)(a3 + 8);
  *((__m128i *)&v26 - 1) = v4;
  *(_QWORD *)v5 = v4.m128i_i64[0];
  *(_DWORD *)(v5 + 8) = *((_DWORD *)&v26 - 2);
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_QWORD *)(v6 + 40);
  if ( *(_DWORD *)(a3 + 16) == *(_DWORD *)(v7 + 24) )
  {
    v12 = *(_DWORD *)(v6 + 48);
    v13 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
    v14 = *(_QWORD *)(a3 + 24);
    *((__m128i *)&v26 - 2) = v13;
    *(_QWORD *)v14 = v13.m128i_i64[0];
    *(_DWORD *)(v14 + 8) = *((_DWORD *)&v26 - 6);
    v15 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v7 + 40) + 40LL));
    v16 = *(_QWORD *)(a3 + 32);
    *((__m128i *)&v26 - 3) = v15;
    *(_QWORD *)v16 = v15.m128i_i64[0];
    *(_DWORD *)(v16 + 8) = *((_DWORD *)&v26 - 10);
    if ( !*(_BYTE *)(a3 + 44) || *(_DWORD *)(a3 + 40) == (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v7 + 28)) )
    {
      v17 = *(_QWORD *)(v7 + 56);
      if ( v17 )
      {
        v18 = 1;
        do
        {
          if ( v12 == *(_DWORD *)(v17 + 8) )
          {
            if ( !v18 )
              goto LABEL_18;
            v17 = *(_QWORD *)(v17 + 32);
            if ( !v17 )
              goto LABEL_20;
            if ( v12 == *(_DWORD *)(v17 + 8) )
              goto LABEL_18;
            v18 = 0;
          }
          v17 = *(_QWORD *)(v17 + 32);
        }
        while ( v17 );
        if ( v18 != 1 )
          goto LABEL_20;
      }
    }
LABEL_18:
    v6 = *(_QWORD *)(a1 + 40);
  }
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v9 = *(_QWORD *)(a3 + 8);
  *((__m128i *)&v26 - 4) = v8;
  *(_QWORD *)v9 = v8.m128i_i64[0];
  *(_DWORD *)(v9 + 8) = *((_DWORD *)&v26 - 14);
  v10 = *(_DWORD **)(a1 + 40);
  v11 = *(_QWORD *)v10;
  if ( *(_DWORD *)(a3 + 16) != *(_DWORD *)(*(_QWORD *)v10 + 24LL) )
    return 0;
  v19 = v10[2];
  v20 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v11 + 40));
  v21 = *(_QWORD *)(a3 + 24);
  *((__m128i *)&v26 - 5) = v20;
  *(_QWORD *)v21 = v20.m128i_i64[0];
  *(_DWORD *)(v21 + 8) = *((_DWORD *)&v26 - 18);
  v22 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v11 + 40) + 40LL));
  v23 = *(_QWORD *)(a3 + 32);
  *((__m128i *)&v26 - 6) = v22;
  *(_QWORD *)v23 = v22.m128i_i64[0];
  *(_DWORD *)(v23 + 8) = *((_DWORD *)&v26 - 22);
  if ( *(_BYTE *)(a3 + 44) )
  {
    if ( *(_DWORD *)(a3 + 40) != (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v11 + 28)) )
      return 0;
  }
  v24 = *(_QWORD *)(v11 + 56);
  if ( !v24 )
    return 0;
  v25 = 1;
  do
  {
    if ( v19 == *(_DWORD *)(v24 + 8) )
    {
      if ( !v25 )
        return 0;
      v24 = *(_QWORD *)(v24 + 32);
      if ( !v24 )
        goto LABEL_20;
      if ( v19 == *(_DWORD *)(v24 + 8) )
        return 0;
      v25 = 0;
    }
    v24 = *(_QWORD *)(v24 + 32);
  }
  while ( v24 );
  if ( v25 == 1 )
    return 0;
LABEL_20:
  result = 1;
  if ( *(_BYTE *)(a3 + 52) )
    return (*(_DWORD *)(a3 + 48) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 48);
  return result;
}
