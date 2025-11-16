// Function: sub_329E540
// Address: 0x329e540
//
bool __fastcall sub_329E540(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i v15; // xmm0
  __int64 v16; // rdx
  __int64 *v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // r8
  __m128i v20; // xmm0
  __int64 v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // rdx
  __m128i v24; // xmm0
  __int64 v25; // rdx
  __int64 v26; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v6 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v26 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v26 - 2);
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_QWORD *)(v7 + 40);
  if ( *(_DWORD *)(a4 + 16) != *(_DWORD *)(v8 + 24) )
  {
LABEL_4:
    v9 = _mm_loadu_si128((const __m128i *)(v7 + 40));
    v10 = *(_QWORD *)(a4 + 8);
    *((__m128i *)&v26 - 4) = v9;
    *(_QWORD *)v10 = v9.m128i_i64[0];
    *(_DWORD *)(v10 + 8) = *((_DWORD *)&v26 - 14);
    v11 = **(_QWORD **)(a1 + 40);
    if ( *(_DWORD *)(a4 + 16) != *(_DWORD *)(v11 + 24) )
      return 0;
    v17 = *(__int64 **)(v11 + 40);
    v18 = *(_QWORD *)(a4 + 24);
    v19 = *v17;
    if ( v18 )
    {
      if ( v19 == v18 && *((_DWORD *)v17 + 2) == *(_DWORD *)(a4 + 32) )
        goto LABEL_31;
      if ( v18 != v17[5] || *(_DWORD *)(a4 + 32) != *((_DWORD *)v17 + 12) )
        return 0;
    }
    else
    {
      if ( v19 )
      {
LABEL_31:
        v24 = _mm_loadu_si128((const __m128i *)(v17 + 5));
        v25 = *(_QWORD *)(a4 + 40);
        *((__m128i *)&v26 - 5) = v24;
        *(_QWORD *)v25 = v24.m128i_i64[0];
        *(_DWORD *)(v25 + 8) = *((_DWORD *)&v26 - 18);
        goto LABEL_21;
      }
      if ( !v17[5] )
        return 0;
    }
    v20 = _mm_loadu_si128((const __m128i *)v17);
    v21 = *(_QWORD *)(a4 + 40);
    *((__m128i *)&v26 - 6) = v20;
    *(_QWORD *)v21 = v20.m128i_i64[0];
    *(_DWORD *)(v21 + 8) = *((_DWORD *)&v26 - 22);
LABEL_21:
    if ( *(_BYTE *)(a4 + 52) && *(_DWORD *)(a4 + 48) != (*(_DWORD *)(a4 + 48) & *(_DWORD *)(v11 + 28)) )
      return 0;
    goto LABEL_14;
  }
  v12 = *(__int64 **)(v8 + 40);
  v13 = *(_QWORD *)(a4 + 24);
  v14 = *v12;
  if ( v13 )
  {
    if ( v14 == v13 && *((_DWORD *)v12 + 2) == *(_DWORD *)(a4 + 32) )
      goto LABEL_25;
    if ( v13 != v12[5] || *(_DWORD *)(a4 + 32) != *((_DWORD *)v12 + 12) )
      goto LABEL_4;
  }
  else
  {
    if ( v14 )
    {
LABEL_25:
      v22 = _mm_loadu_si128((const __m128i *)(v12 + 5));
      v23 = *(_QWORD *)(a4 + 40);
      *((__m128i *)&v26 - 2) = v22;
      *(_QWORD *)v23 = v22.m128i_i64[0];
      *(_DWORD *)(v23 + 8) = *((_DWORD *)&v26 - 6);
      goto LABEL_12;
    }
    if ( !v12[5] )
      goto LABEL_4;
  }
  v15 = _mm_loadu_si128((const __m128i *)v12);
  v16 = *(_QWORD *)(a4 + 40);
  *((__m128i *)&v26 - 3) = v15;
  *(_QWORD *)v16 = v15.m128i_i64[0];
  *(_DWORD *)(v16 + 8) = *((_DWORD *)&v26 - 10);
LABEL_12:
  if ( *(_BYTE *)(a4 + 52) && *(_DWORD *)(a4 + 48) != (*(_DWORD *)(a4 + 48) & *(_DWORD *)(v8 + 28)) )
  {
    v7 = *(_QWORD *)(a1 + 40);
    goto LABEL_4;
  }
LABEL_14:
  result = 1;
  if ( *(_BYTE *)(a4 + 60) )
    return (*(_DWORD *)(a4 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 56);
  return result;
}
