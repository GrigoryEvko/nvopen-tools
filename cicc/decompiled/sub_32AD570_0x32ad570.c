// Function: sub_32AD570
// Address: 0x32ad570
//
__int64 __fastcall sub_32AD570(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __m128i v5; // xmm0
  __int64 v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rax
  __int64 v9; // rax
  _DWORD *v10; // rdx
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // [rsp-8h] [rbp-8h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v6 = *(_QWORD *)(a4 + 8);
  *((__m128i *)&v14 - 1) = v5;
  *(_QWORD *)v6 = v5.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = *((_DWORD *)&v14 - 2);
  v7 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
  v8 = *(_QWORD *)(a4 + 16);
  *((__m128i *)&v14 - 2) = v7;
  *(_QWORD *)v8 = v7.m128i_i64[0];
  *(_DWORD *)(v8 + 8) = *((_DWORD *)&v14 - 6);
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 80LL);
  if ( *(_DWORD *)(v9 + 24) != 8 || *(_BYTE *)(a4 + 28) && *(_DWORD *)(a4 + 24) != *(_DWORD *)(v9 + 96) )
    return 0;
  v10 = *(_DWORD **)(a4 + 32);
  if ( v10 )
    *v10 = *(_DWORD *)(v9 + 96);
  v11 = *(_QWORD *)(a1 + 56);
  if ( !v11 )
    return 0;
  v12 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v11 + 8) != a2 )
    {
      v11 = *(_QWORD *)(v11 + 32);
      if ( !v11 )
        return v12 ^ 1u;
    }
    if ( !v12 )
      return 0;
    v13 = *(_QWORD *)(v11 + 32);
    if ( !v13 )
      break;
    if ( a2 == *(_DWORD *)(v13 + 8) )
      return 0;
    v11 = *(_QWORD *)(v13 + 32);
    v12 = 0;
    if ( !v11 )
      return v12 ^ 1u;
  }
  return 1;
}
