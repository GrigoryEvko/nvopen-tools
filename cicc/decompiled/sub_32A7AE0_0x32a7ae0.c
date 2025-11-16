// Function: sub_32A7AE0
// Address: 0x32a7ae0
//
bool __fastcall sub_32A7AE0(__int64 a1, __int64 a2, __int64 a3)
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
  __int64 v15; // rax
  int v16; // esi
  int v17; // ecx
  __m128i v18; // xmm0
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // esi
  __int64 v22; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v4 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v5 = *(_QWORD *)(a3 + 8);
  *((__m128i *)&v22 - 1) = v4;
  *(_QWORD *)v5 = v4.m128i_i64[0];
  *(_DWORD *)(v5 + 8) = *((_DWORD *)&v22 - 2);
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_QWORD *)(v6 + 40);
  if ( *(_DWORD *)(a3 + 16) == *(_DWORD *)(v7 + 24) )
  {
    v12 = *(_DWORD *)(v6 + 48);
    v13 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
    v14 = *(_QWORD *)(a3 + 24);
    *((__m128i *)&v22 - 2) = v13;
    *(_QWORD *)v14 = v13.m128i_i64[0];
    *(_DWORD *)(v14 + 8) = *((_DWORD *)&v22 - 6);
    if ( !*(_BYTE *)(a3 + 36) || *(_DWORD *)(a3 + 32) == (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v7 + 28)) )
    {
      v15 = *(_QWORD *)(v7 + 56);
      if ( v15 )
      {
        v16 = 1;
        do
        {
          if ( v12 == *(_DWORD *)(v15 + 8) )
          {
            if ( !v16 )
              goto LABEL_18;
            v15 = *(_QWORD *)(v15 + 32);
            if ( !v15 )
              goto LABEL_20;
            if ( v12 == *(_DWORD *)(v15 + 8) )
              goto LABEL_18;
            v16 = 0;
          }
          v15 = *(_QWORD *)(v15 + 32);
        }
        while ( v15 );
        if ( v16 != 1 )
          goto LABEL_20;
      }
    }
LABEL_18:
    v6 = *(_QWORD *)(a1 + 40);
  }
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v9 = *(_QWORD *)(a3 + 8);
  *((__m128i *)&v22 - 3) = v8;
  *(_QWORD *)v9 = v8.m128i_i64[0];
  *(_DWORD *)(v9 + 8) = *((_DWORD *)&v22 - 10);
  v10 = *(_DWORD **)(a1 + 40);
  v11 = *(_QWORD *)v10;
  if ( *(_DWORD *)(a3 + 16) != *(_DWORD *)(*(_QWORD *)v10 + 24LL) )
    return 0;
  v17 = v10[2];
  v18 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v11 + 40));
  v19 = *(_QWORD *)(a3 + 24);
  *((__m128i *)&v22 - 4) = v18;
  *(_QWORD *)v19 = v18.m128i_i64[0];
  *(_DWORD *)(v19 + 8) = *((_DWORD *)&v22 - 14);
  if ( *(_BYTE *)(a3 + 36) )
  {
    if ( *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v11 + 28)) )
      return 0;
  }
  v20 = *(_QWORD *)(v11 + 56);
  if ( !v20 )
    return 0;
  v21 = 1;
  do
  {
    if ( v17 == *(_DWORD *)(v20 + 8) )
    {
      if ( !v21 )
        return 0;
      v20 = *(_QWORD *)(v20 + 32);
      if ( !v20 )
        goto LABEL_20;
      if ( *(_DWORD *)(v20 + 8) == v17 )
        return 0;
      v21 = 0;
    }
    v20 = *(_QWORD *)(v20 + 32);
  }
  while ( v20 );
  if ( v21 == 1 )
    return 0;
LABEL_20:
  result = 1;
  if ( *(_BYTE *)(a3 + 44) )
    return (*(_DWORD *)(a3 + 40) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 40);
  return result;
}
