// Function: sub_32A63F0
// Address: 0x32a63f0
//
bool __fastcall sub_32A63F0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 v4; // rax
  __m128i v5; // xmm0
  __int64 v6; // rcx
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rcx
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v4 = **(_QWORD **)(a1 + 40);
  if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(v4 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v4 + 40));
  v6 = *(_QWORD *)(a3 + 16);
  *((__m128i *)&v10 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v10 - 2);
  if ( *(_BYTE *)(a3 + 28) && *(_DWORD *)(a3 + 24) != (*(_DWORD *)(a3 + 24) & *(_DWORD *)(v4 + 28)) )
    return 0;
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
  if ( *(_DWORD *)(a3 + 32) != *(_DWORD *)(v7 + 24) )
    return 0;
  v8 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
  v9 = *(_QWORD *)(a3 + 40);
  *((__m128i *)&v10 - 2) = v8;
  *(_QWORD *)v9 = v8.m128i_i64[0];
  *(_DWORD *)(v9 + 8) = *((_DWORD *)&v10 - 6);
  if ( *(_BYTE *)(a3 + 52) )
  {
    if ( *(_DWORD *)(a3 + 48) != (*(_DWORD *)(a3 + 48) & *(_DWORD *)(v7 + 28)) )
      return 0;
  }
  result = 1;
  if ( *(_BYTE *)(a3 + 60) )
    return (*(_DWORD *)(a3 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 56);
  return result;
}
