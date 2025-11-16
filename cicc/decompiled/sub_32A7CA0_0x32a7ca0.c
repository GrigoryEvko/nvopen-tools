// Function: sub_32A7CA0
// Address: 0x32a7ca0
//
bool __fastcall sub_32A7CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __int64 v6; // rax
  __int64 v7; // rax
  __m128i v8; // [rsp-38h] [rbp-38h]
  __m128i v9; // [rsp-28h] [rbp-28h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v6 = *(_QWORD *)(a4 + 8);
  v9 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  *(_QWORD *)v6 = v9.m128i_i64[0];
  *(_DWORD *)(v6 + 8) = v9.m128i_i32[2];
  if ( !(unsigned __int8)sub_33E07E0(
                           *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL),
                           0) )
  {
    v7 = *(_QWORD *)(a4 + 8);
    v8 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
    *(_QWORD *)v7 = v8.m128i_i64[0];
    *(_DWORD *)(v7 + 8) = v8.m128i_i32[2];
    if ( !(unsigned __int8)sub_33E07E0(**(_QWORD **)(a1 + 40), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL), 0) )
      return 0;
  }
  result = 1;
  if ( *(_BYTE *)(a4 + 24) )
    return (*(_DWORD *)(a4 + 20) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 20);
  return result;
}
