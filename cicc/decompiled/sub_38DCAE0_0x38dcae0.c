// Function: sub_38DCAE0
// Address: 0x38dcae0
//
__int64 __fastcall sub_38DCAE0(__m128i *a1, __int64 a2)
{
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __m128i v5; // [rsp+0h] [rbp-20h] BYREF
  __m128i v6; // [rsp+10h] [rbp-10h] BYREF

  a1->m128i_i64[1] = a2;
  v5 = 0u;
  v6 = 0u;
  v2 = _mm_loadu_si128(&v5);
  v3 = _mm_loadu_si128(&v6);
  a1->m128i_i64[0] = (__int64)&unk_4A3E670;
  a1[7].m128i_i64[0] = (__int64)a1[8].m128i_i64;
  a1[1].m128i_i64[0] = 0;
  a1[1].m128i_i64[1] = 0;
  a1[2].m128i_i64[0] = 0;
  a1[2].m128i_i64[1] = 0;
  a1[3].m128i_i64[0] = 0;
  a1[3].m128i_i64[1] = 0;
  a1[4].m128i_i64[0] = 0;
  a1[4].m128i_i64[1] = 0;
  a1[5].m128i_i64[0] = 0;
  a1[5].m128i_i64[1] = 0;
  a1[6].m128i_i64[0] = 0;
  a1[6].m128i_i32[2] = 0;
  a1[16].m128i_i32[0] = 0;
  a1[16].m128i_i8[4] = 0;
  a1[7].m128i_i64[1] = 0x400000001LL;
  a1[8] = v2;
  a1[9] = v3;
  return 0x400000001LL;
}
