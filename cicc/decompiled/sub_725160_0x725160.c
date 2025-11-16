// Function: sub_725160
// Address: 0x725160
//
__m128i *sub_725160()
{
  __m128i *result; // rax
  __int64 v1; // rdx
  __m128i v2; // xmm0

  result = (__m128i *)sub_7247C0(152);
  result->m128i_i64[0] = 0;
  v1 = *(_QWORD *)&dword_4F077C8;
  result->m128i_i64[1] = 0;
  result[6].m128i_i32[0] &= 0xFE00u;
  result[4].m128i_i64[1] = v1;
  result[1].m128i_i64[0] = 0;
  v2 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  result[1].m128i_i64[1] = 0;
  result[2].m128i_i64[0] = 0;
  result[2].m128i_i64[1] = 0;
  result[3].m128i_i64[0] = 0;
  result[3].m128i_i64[1] = 0;
  result[4].m128i_i64[0] = 0;
  result[6].m128i_i64[1] = 0;
  result[7].m128i_i64[0] = 0;
  result[7].m128i_i64[1] = 0;
  result[8].m128i_i64[0] = -1;
  result[8].m128i_i32[2] = 0;
  result[9].m128i_i64[0] = 0;
  result[5] = v2;
  return result;
}
