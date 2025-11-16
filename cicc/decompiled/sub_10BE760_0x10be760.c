// Function: sub_10BE760
// Address: 0x10be760
//
__int64 __fastcall sub_10BE760(__int64 a1, const __m128i *a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  unsigned __int64 v7; // xmm2_8
  __m128i v8; // xmm3
  __m128i v10[2]; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v11; // [rsp+20h] [rbp-40h]
  __int64 v12; // [rsp+28h] [rbp-38h]
  __m128i v13; // [rsp+30h] [rbp-30h]
  __int64 v14; // [rsp+40h] [rbp-20h]

  v5 = _mm_loadu_si128(a2 + 6);
  v6 = _mm_loadu_si128(a2 + 7);
  v7 = _mm_loadu_si128(a2 + 8).m128i_u64[0];
  v8 = _mm_loadu_si128(a2 + 9);
  v14 = a2[10].m128i_i64[0];
  v11 = v7;
  v10[0] = v5;
  v12 = a5;
  v10[1] = v6;
  v13 = v8;
  sub_9AC330(a1, a3, a4, v10);
  return a1;
}
