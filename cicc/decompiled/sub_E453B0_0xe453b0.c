// Function: sub_E453B0
// Address: 0xe453b0
//
__m128i *__fastcall sub_E453B0(__m128i *a1, __int64 a2, __int64 a3)
{
  __m128i *result; // rax
  bool v4; // zf
  __int64 v5; // [rsp-8h] [rbp-8h] BYREF

  result = a1;
  if ( a3 )
  {
    v4 = (*(_BYTE *)(a3 + 7) & 0x10) == 0;
    a1->m128i_i64[0] = a3;
    if ( v4 )
    {
      a1[1].m128i_i64[0] = (__int64)sub_E45100;
      a1[1].m128i_i64[1] = (__int64)sub_E45450;
    }
    else
    {
      a1[1].m128i_i64[0] = (__int64)sub_E450D0;
      a1[1].m128i_i64[1] = (__int64)sub_E45160;
    }
  }
  else
  {
    a1[1].m128i_i64[0] = (__int64)sub_E45070;
    a1[1].m128i_i64[1] = (__int64)sub_E451C0;
    *a1 = _mm_loadu_si128((const __m128i *)&v5 - 2);
  }
  return result;
}
