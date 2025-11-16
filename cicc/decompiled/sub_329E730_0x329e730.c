// Function: sub_329E730
// Address: 0x329e730
//
bool __fastcall sub_329E730(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // r8
  __int64 v13; // rsi
  __int64 v14; // r9
  __m128i v15; // xmm0
  __int64 v16; // rdx
  __int64 *v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r8
  __m128i v20; // xmm0
  __int64 v21; // rdx
  __int64 v22; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v6 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v22 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v22 - 2);
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_QWORD *)(v7 + 40);
  if ( *(_DWORD *)(a4 + 16) != *(_DWORD *)(v8 + 24) )
    goto LABEL_4;
  v12 = *(__int64 **)(v8 + 40);
  v13 = *(_QWORD *)(a4 + 24);
  v14 = *v12;
  if ( v13 )
  {
    if ( v14 != v13 || *((_DWORD *)v12 + 2) != *(_DWORD *)(a4 + 32) )
      goto LABEL_4;
  }
  else if ( !v14 )
  {
    goto LABEL_4;
  }
  v15 = _mm_loadu_si128((const __m128i *)(v12 + 5));
  v16 = *(_QWORD *)(a4 + 40);
  *((__m128i *)&v22 - 2) = v15;
  *(_QWORD *)v16 = v15.m128i_i64[0];
  *(_DWORD *)(v16 + 8) = *((_DWORD *)&v22 - 6);
  if ( !*(_BYTE *)(a4 + 52) || *(_DWORD *)(a4 + 48) == (*(_DWORD *)(a4 + 48) & *(_DWORD *)(v8 + 28)) )
    goto LABEL_12;
  v7 = *(_QWORD *)(a1 + 40);
LABEL_4:
  v9 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v10 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v22 - 3) = v9;
  *(_QWORD *)v10 = v9.m128i_i64[0];
  *(_DWORD *)(v10 + 8) = *((_DWORD *)&v22 - 10);
  v11 = **(_QWORD **)(a1 + 40);
  if ( *(_DWORD *)(a4 + 16) != *(_DWORD *)(v11 + 24) )
    return 0;
  v17 = *(__int64 **)(v11 + 40);
  v18 = *(_QWORD *)(a4 + 24);
  v19 = *v17;
  if ( v18 )
  {
    if ( v19 != v18 || *((_DWORD *)v17 + 2) != *(_DWORD *)(a4 + 32) )
      return 0;
  }
  else if ( !v19 )
  {
    return 0;
  }
  v20 = _mm_loadu_si128((const __m128i *)(v17 + 5));
  v21 = *(_QWORD *)(a4 + 40);
  *((__m128i *)&v22 - 4) = v20;
  *(_QWORD *)v21 = v20.m128i_i64[0];
  *(_DWORD *)(v21 + 8) = *((_DWORD *)&v22 - 14);
  if ( *(_BYTE *)(a4 + 52) && *(_DWORD *)(a4 + 48) != (*(_DWORD *)(a4 + 48) & *(_DWORD *)(v11 + 28)) )
    return 0;
LABEL_12:
  result = 1;
  if ( *(_BYTE *)(a4 + 60) )
    return (*(_DWORD *)(a4 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 56);
  return result;
}
