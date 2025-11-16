// Function: sub_32A8450
// Address: 0x32a8450
//
__int64 __fastcall sub_32A8450(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rax
  __m128i v11; // [rsp-48h] [rbp-48h]
  __m128i v12; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v6 = *(_QWORD *)(a4 + 8);
  v12 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  *(_QWORD *)v6 = v12.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = v12.m128i_i32[2];
  if ( !(unsigned __int8)sub_33E07E0(
                           *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL),
                           0) )
  {
    v10 = *(_QWORD *)(a4 + 8);
    v11 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
    *(_QWORD *)v10 = v11.m128i_i64[0];
    *(_DWORD *)(v10 + 8) = v11.m128i_i32[2];
    if ( !(unsigned __int8)sub_33E07E0(**(_QWORD **)(a1 + 40), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL), 0) )
      return 0;
  }
  if ( *(_BYTE *)(a4 + 24) && *(_DWORD *)(a4 + 20) != (*(_DWORD *)(a4 + 20) & *(_DWORD *)(a1 + 28)) )
    return 0;
  v7 = *(_QWORD *)(a1 + 56);
  if ( !v7 )
    return 0;
  v8 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v7 + 8) != a2 )
    {
      v7 = *(_QWORD *)(v7 + 32);
      if ( !v7 )
        return v8 ^ 1u;
    }
    if ( !v8 )
      return 0;
    v9 = *(_QWORD *)(v7 + 32);
    if ( !v9 )
      break;
    if ( a2 == *(_DWORD *)(v9 + 8) )
      return 0;
    v7 = *(_QWORD *)(v9 + 32);
    v8 = 0;
    if ( !v7 )
      return v8 ^ 1u;
  }
  return 1;
}
