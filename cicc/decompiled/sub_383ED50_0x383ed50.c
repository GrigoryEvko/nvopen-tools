// Function: sub_383ED50
// Address: 0x383ed50
//
__int64 *__fastcall sub_383ED50(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // zf
  __m128i v4; // xmm0
  __int32 v6; // edx
  __int32 v7; // edx
  __m128i v8; // [rsp+20h] [rbp-30h] BYREF
  __m128i v9; // [rsp+30h] [rbp-20h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_DWORD *)(a2 + 24) == 184;
  v4 = _mm_loadu_si128((const __m128i *)v2);
  v8 = v4;
  v9 = _mm_loadu_si128((const __m128i *)(v2 + 40));
  if ( v3 )
  {
    v8.m128i_i64[0] = (__int64)sub_383B380(a1, v8.m128i_u64[0], v8.m128i_i64[1]);
    v8.m128i_i32[2] = v6;
    v9.m128i_i64[0] = (__int64)sub_383B380(a1, v9.m128i_u64[0], v9.m128i_i64[1]);
    v9.m128i_i32[2] = v7;
  }
  else
  {
    sub_383E4F0((__int64 *)a1, (__int64)&v8, (__int64)&v9, v4);
  }
  return sub_33EC010(
           *(_QWORD **)(a1 + 8),
           (__int64 *)a2,
           v8.m128i_u64[0],
           v8.m128i_u64[1],
           v9.m128i_i64[0],
           v9.m128i_i64[1]);
}
