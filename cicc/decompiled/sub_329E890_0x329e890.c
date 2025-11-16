// Function: sub_329E890
// Address: 0x329e890
//
bool __fastcall sub_329E890(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v6 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v13 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v13 - 2);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
  if ( *(_DWORD *)(a4 + 16) != *(_DWORD *)(v7 + 24) )
    return 0;
  v8 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
  v9 = *(_QWORD *)(a4 + 24);
  *((__m128i *)&v13 - 2) = v8;
  *(_QWORD *)v9 = v8.m128i_i64[0];
  *(_DWORD *)(v9 + 8) = *((_DWORD *)&v13 - 6);
  v10 = *(_QWORD *)(v7 + 40);
  v11 = *(_QWORD *)(a4 + 32);
  v12 = *(_QWORD *)(v10 + 40);
  if ( v11 )
  {
    if ( v12 != v11 || *(_DWORD *)(v10 + 48) != *(_DWORD *)(a4 + 40) )
      return 0;
  }
  else if ( !v12 )
  {
    return 0;
  }
  if ( *(_BYTE *)(a4 + 52) && *(_DWORD *)(a4 + 48) != (*(_DWORD *)(a4 + 48) & *(_DWORD *)(v7 + 28)) )
    return 0;
  result = 1;
  if ( *(_BYTE *)(a4 + 60) )
    return (*(_DWORD *)(a4 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 56);
  return result;
}
