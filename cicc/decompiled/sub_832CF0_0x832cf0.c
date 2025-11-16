// Function: sub_832CF0
// Address: 0x832cf0
//
__int64 __fastcall sub_832CF0(const __m128i *a1, __m128i *a2)
{
  __int64 result; // rax

  if ( a2->m128i_i32[2] == 7 || (result = sub_828DA0((__int64)a1, (__int64)a2), (int)result < 0) )
  {
    *a2 = _mm_loadu_si128(a1);
    a2[1] = _mm_loadu_si128(a1 + 1);
    a2[2] = _mm_loadu_si128(a1 + 2);
    a2[3] = _mm_loadu_si128(a1 + 3);
    a2[4] = _mm_loadu_si128(a1 + 4);
    a2[5] = _mm_loadu_si128(a1 + 5);
    result = a1[6].m128i_i64[0];
    a2[6].m128i_i64[0] = result;
  }
  return result;
}
