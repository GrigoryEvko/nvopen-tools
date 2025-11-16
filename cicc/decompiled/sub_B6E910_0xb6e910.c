// Function: sub_B6E910
// Address: 0xb6e910
//
__int64 __fastcall sub_B6E910(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __m128i v4; // [rsp+0h] [rbp-10h] BYREF

  v4.m128i_i64[0] = a2;
  v4.m128i_i64[1] = a3;
  result = *a1;
  *(__m128i *)(*a1 + 120) = _mm_loadu_si128(&v4);
  return result;
}
