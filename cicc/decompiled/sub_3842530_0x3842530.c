// Function: sub_3842530
// Address: 0x3842530
//
__int64 *__fastcall sub_3842530(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __m128i v3; // xmm0
  __m128i v4; // xmm1
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int128 *v7; // rax
  __int32 v9; // edx
  __int32 v10; // edx
  __m128i v11; // [rsp+20h] [rbp-30h] BYREF
  __m128i v12; // [rsp+30h] [rbp-20h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v3 = _mm_loadu_si128((const __m128i *)v2);
  v4 = _mm_loadu_si128((const __m128i *)(v2 + 40));
  v5 = *(_QWORD *)(v2 + 80);
  v11 = v3;
  LODWORD(v5) = *(_DWORD *)(v5 + 96);
  v12 = v4;
  if ( (unsigned int)(v5 - 18) <= 3 )
  {
    v11.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v11.m128i_u64[0], v11.m128i_i64[1]);
    v11.m128i_i32[2] = v9;
    v12.m128i_i64[0] = (__int64)sub_383B380((__int64)a1, v12.m128i_u64[0], v12.m128i_i64[1]);
    v12.m128i_i32[2] = v10;
  }
  else
  {
    sub_383E4F0(a1, (__int64)&v11, (__int64)&v12, v3);
  }
  v6 = (_QWORD *)a1[1];
  v7 = *(__int128 **)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 208 )
    return sub_33EC3B0(v6, (__int64 *)a2, v11.m128i_i64[0], v11.m128i_i64[1], v12.m128i_i64[0], v12.m128i_i64[1], v7[5]);
  else
    return sub_33EC430(
             v6,
             (__int64 *)a2,
             v11.m128i_i64[0],
             v11.m128i_i64[1],
             v12.m128i_i64[0],
             v12.m128i_i64[1],
             v7[5],
             *(__int128 *)((char *)v7 + 120),
             v7[10]);
}
