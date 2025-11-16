// Function: sub_32AD1B0
// Address: 0x32ad1b0
//
bool __fastcall sub_32AD1B0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 v4; // rax
  __int64 v5; // rcx
  __m128i v6; // xmm0
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __int64 v9; // rcx
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v4 = **(_QWORD **)(a1 + 40);
  if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(v4 + 24) )
    return 0;
  v5 = **(_QWORD **)(v4 + 40);
  if ( *(_DWORD *)(a3 + 16) != *(_DWORD *)(v5 + 24) )
    return 0;
  v6 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v5 + 40));
  v7 = *(_QWORD *)(a3 + 24);
  *((__m128i *)&v10 - 1) = v6;
  *(_QWORD *)v7 = v6.m128i_i64[0];
  *(_DWORD *)(v7 + 8) = *((_DWORD *)&v10 - 2);
  if ( *(_BYTE *)(a3 + 36) && *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v5 + 28)) )
    return 0;
  v8 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v4 + 40) + 40LL));
  v9 = *(_QWORD *)(a3 + 40);
  *((__m128i *)&v10 - 2) = v8;
  *(_QWORD *)v9 = v8.m128i_i64[0];
  *(_DWORD *)(v9 + 8) = *((_DWORD *)&v10 - 6);
  if ( *(_BYTE *)(a3 + 52) )
  {
    if ( *(_DWORD *)(a3 + 48) != (*(_DWORD *)(a3 + 48) & *(_DWORD *)(v4 + 28)) )
      return 0;
  }
  result = 1;
  if ( *(_BYTE *)(a3 + 60) )
    return (*(_DWORD *)(a3 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 56);
  return result;
}
