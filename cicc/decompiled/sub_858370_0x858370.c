// Function: sub_858370
// Address: 0x858370
//
const __m128i *__fastcall sub_858370(__int64 *a1)
{
  const __m128i *result; // rax
  __int64 *v2; // r12
  __m128i *v3; // rbx
  __m128i v4; // xmm4

  result = (const __m128i *)qword_4D03CB8;
  if ( qword_4D03CB8[1] )
  {
    v2 = a1;
    v3 = (__m128i *)sub_727670();
    result = (const __m128i *)qword_4D03CB8[1];
    *v3 = _mm_loadu_si128(result);
    v3[1] = _mm_loadu_si128(result + 1);
    v3[2] = _mm_loadu_si128(result + 2);
    v3[3] = _mm_loadu_si128(result + 3);
    v4 = _mm_loadu_si128(result + 4);
    v3->m128i_i64[0] = 0;
    v3[4] = v4;
    if ( !a1 )
    {
      MEMORY[0] = v3;
      BUG();
    }
    if ( *a1 )
    {
      result = (const __m128i *)sub_5CB9F0((_QWORD **)a1);
      v2 = (__int64 *)result;
    }
    *v2 = (__int64)v3;
  }
  return result;
}
