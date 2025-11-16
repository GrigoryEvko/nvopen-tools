// Function: sub_329D420
// Address: 0x329d420
//
bool __fastcall sub_329D420(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r8
  __m128i v11; // xmm0
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __int64 v14; // rdx
  __int64 v15; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v6 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v15 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v15 - 2);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
  if ( *(_DWORD *)(a4 + 16) != *(_DWORD *)(v7 + 24) )
    return 0;
  v8 = *(__int64 **)(v7 + 40);
  v9 = *(_QWORD *)(a4 + 24);
  v10 = *v8;
  if ( v9 )
  {
    if ( v10 == v9 && *((_DWORD *)v8 + 2) == *(_DWORD *)(a4 + 32) )
      goto LABEL_16;
    if ( v9 != v8[5] || *(_DWORD *)(a4 + 32) != *((_DWORD *)v8 + 12) )
      return 0;
  }
  else
  {
    if ( v10 )
    {
LABEL_16:
      v13 = _mm_loadu_si128((const __m128i *)(v8 + 5));
      v14 = *(_QWORD *)(a4 + 40);
      *((__m128i *)&v15 - 2) = v13;
      *(_QWORD *)v14 = v13.m128i_i64[0];
      *(_DWORD *)(v14 + 8) = *((_DWORD *)&v15 - 6);
      goto LABEL_11;
    }
    if ( !v8[5] )
      return 0;
  }
  v11 = _mm_loadu_si128((const __m128i *)v8);
  v12 = *(_QWORD *)(a4 + 40);
  *((__m128i *)&v15 - 3) = v11;
  *(_QWORD *)v12 = v11.m128i_i64[0];
  *(_DWORD *)(v12 + 8) = *((_DWORD *)&v15 - 10);
LABEL_11:
  if ( *(_BYTE *)(a4 + 52) && *(_DWORD *)(a4 + 48) != (*(_DWORD *)(a4 + 48) & *(_DWORD *)(v7 + 28)) )
    return 0;
  result = 1;
  if ( *(_BYTE *)(a4 + 60) )
    return (*(_DWORD *)(a4 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 56);
  return result;
}
