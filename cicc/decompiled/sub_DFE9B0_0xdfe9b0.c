// Function: sub_DFE9B0
// Address: 0xdfe9b0
//
__int64 __fastcall sub_DFE9B0(__m128i *a1, __m128i *a2)
{
  __m128i v2; // xmm1
  __int64 v3; // rdx
  __int64 v4; // rax
  __m128i v5; // xmm0
  __int64 result; // rax

  v2 = _mm_loadu_si128(a1);
  v3 = a1[1].m128i_i64[1];
  a1[1].m128i_i64[0] = 0;
  v4 = a2[1].m128i_i64[0];
  v5 = _mm_loadu_si128(a2);
  a2[1].m128i_i64[0] = 0;
  *a2 = v2;
  a1[1].m128i_i64[0] = v4;
  result = a2[1].m128i_i64[1];
  *a1 = v5;
  a2[1].m128i_i64[1] = v3;
  a1[1].m128i_i64[1] = result;
  return result;
}
