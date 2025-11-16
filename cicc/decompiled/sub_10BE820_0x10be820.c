// Function: sub_10BE820
// Address: 0x10be820
//
__int64 __fastcall sub_10BE820(const __m128i *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __m128i v5; // xmm1
  unsigned __int64 v6; // xmm2_8
  __m128i v7; // xmm3
  __int64 v8; // rax
  __m128i v10[2]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v11; // [rsp+20h] [rbp-30h]
  __int64 v12; // [rsp+28h] [rbp-28h]
  __m128i v13; // [rsp+30h] [rbp-20h]
  __int64 v14; // [rsp+40h] [rbp-10h]

  v5 = _mm_loadu_si128(a1 + 7);
  v6 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v7 = _mm_loadu_si128(a1 + 9);
  v8 = a1[10].m128i_i64[0];
  v10[0] = _mm_loadu_si128(a1 + 6);
  v11 = v6;
  v14 = v8;
  v12 = a5;
  v10[1] = v5;
  v13 = v7;
  return sub_9AC230(a2, a3, v10, a4);
}
