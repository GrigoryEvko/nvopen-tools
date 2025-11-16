// Function: sub_329D3B0
// Address: 0x329d3b0
//
bool __fastcall sub_329D3B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __m128i v5; // xmm0
  __int64 v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rax
  __int64 v9; // [rsp-8h] [rbp-8h]

  result = 0;
  if ( *(_DWORD *)a4 == *(_DWORD *)(a1 + 24) )
  {
    v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
    v6 = *(_QWORD *)(a4 + 8);
    *((__m128i *)&v9 - 1) = v5;
    *(_QWORD *)v6 = v5.m128i_i64[0];
    *(_DWORD *)(v6 + 8) = *((_DWORD *)&v9 - 2);
    v7 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
    v8 = *(_QWORD *)(a4 + 16);
    *((__m128i *)&v9 - 2) = v7;
    *(_QWORD *)v8 = v7.m128i_i64[0];
    *(_DWORD *)(v8 + 8) = *((_DWORD *)&v9 - 6);
    result = 1;
    if ( *(_BYTE *)(a4 + 28) )
      return (*(_DWORD *)(a4 + 24) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 24);
  }
  return result;
}
