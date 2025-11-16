// Function: sub_329D770
// Address: 0x329d770
//
__int64 __fastcall sub_329D770(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __m128i v5; // xmm0
  __int64 v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v6 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v12 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v12 - 2);
  v7 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
  v8 = *(_QWORD *)(a4 + 16);
  *((__m128i *)&v12 - 2) = v7;
  *(_QWORD *)v8 = v7.m128i_i64[0];
  *(_DWORD *)(v8 + 8) = *((_DWORD *)&v12 - 6);
  if ( *(_BYTE *)(a4 + 28) && *(_DWORD *)(a4 + 24) != (*(_DWORD *)(a4 + 24) & *(_DWORD *)(a1 + 28)) )
    return 0;
  v9 = *(_QWORD *)(a1 + 56);
  if ( !v9 )
    return 0;
  v10 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v9 + 8) != a2 )
    {
      v9 = *(_QWORD *)(v9 + 32);
      if ( !v9 )
        return v10 ^ 1u;
    }
    if ( !v10 )
      return 0;
    v11 = *(_QWORD *)(v9 + 32);
    if ( !v11 )
      break;
    if ( a2 == *(_DWORD *)(v11 + 8) )
      return 0;
    v9 = *(_QWORD *)(v11 + 32);
    v10 = 0;
    if ( !v9 )
      return v10 ^ 1u;
  }
  return 1;
}
