// Function: sub_2EE72D0
// Address: 0x2ee72d0
//
__m128i *__fastcall sub_2EE72D0(__m128i *a1, __int64 a2, __int64 a3)
{
  __m128i *result; // rax
  __int64 v4; // [rsp-8h] [rbp-8h] BYREF

  result = a1;
  if ( a3 )
  {
    a1->m128i_i64[0] = a3;
    a1[1].m128i_i64[0] = (__int64)sub_2EE6D70;
    a1[1].m128i_i64[1] = (__int64)sub_2EE6D50;
  }
  else
  {
    a1[1].m128i_i64[0] = (__int64)sub_2EE6D40;
    a1[1].m128i_i64[1] = (__int64)sub_2EE6F70;
    *a1 = _mm_loadu_si128((const __m128i *)&v4 - 2);
  }
  return result;
}
