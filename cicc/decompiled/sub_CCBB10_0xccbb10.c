// Function: sub_CCBB10
// Address: 0xccbb10
//
__int64 __fastcall sub_CCBB10(__m128i *a1, const __m128i *a2)
{
  unsigned __int64 v3; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    a1->m128i_i64[1] = 0;
    v3 = (unsigned __int64)&a1[1];
    *(_QWORD *)(v3 + 224) = 0;
    memset((void *)(v3 & 0xFFFFFFFFFFFFFFF8LL), 0, 8 * (((unsigned int)a1 - (v3 & 0xFFFFFFF8) + 248) >> 3));
    a1->m128i_i32[2] = 2;
    a1->m128i_i64[0] = 0x10000012CLL;
    a1[1].m128i_i8[4] |= 1u;
    a1[5] = _mm_loadu_si128(a2);
    a1[6] = _mm_loadu_si128(a2 + 1);
    a1[7] = _mm_loadu_si128(a2 + 2);
    a1[8] = _mm_loadu_si128(a2 + 3);
    a1[9] = _mm_loadu_si128(a2 + 4);
    a1[10] = _mm_loadu_si128(a2 + 5);
    a1[11] = _mm_loadu_si128(a2 + 6);
    result = a2[7].m128i_i64[0];
    a1[12].m128i_i64[0] = result;
  }
  return result;
}
