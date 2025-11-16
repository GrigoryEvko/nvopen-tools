// Function: sub_3842450
// Address: 0x3842450
//
__int64 *__fastcall sub_3842450(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __m128i v3; // xmm0
  __m128i v4; // xmm1
  __int64 v5; // rax
  __int32 v7; // edx
  __int32 v8; // edx
  __m128i v9; // [rsp+20h] [rbp-30h] BYREF
  __m128i v10; // [rsp+30h] [rbp-20h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v3 = _mm_loadu_si128((const __m128i *)(v2 + 80));
  v4 = _mm_loadu_si128((const __m128i *)(v2 + 120));
  v5 = *(_QWORD *)(v2 + 40);
  v9 = v3;
  LODWORD(v5) = *(_DWORD *)(v5 + 96);
  v10 = v4;
  if ( (unsigned int)(v5 - 18) <= 3 )
  {
    v9.m128i_i64[0] = (__int64)sub_383B380(a1, v9.m128i_u64[0], v9.m128i_i64[1]);
    v9.m128i_i32[2] = v7;
    v10.m128i_i64[0] = (__int64)sub_383B380(a1, v10.m128i_u64[0], v10.m128i_i64[1]);
    v10.m128i_i32[2] = v8;
  }
  else
  {
    sub_383E4F0((__int64 *)a1, (__int64)&v9, (__int64)&v10, v3);
  }
  return sub_33EC430(
           *(_QWORD **)(a1 + 8),
           (__int64 *)a2,
           **(_QWORD **)(a2 + 40),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
           *(_OWORD *)&v9,
           *(_OWORD *)&v10,
           *(_OWORD *)(*(_QWORD *)(a2 + 40) + 160LL));
}
