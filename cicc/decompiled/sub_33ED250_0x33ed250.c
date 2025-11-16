// Function: sub_33ED250
// Address: 0x33ed250
//
__m128i *__fastcall sub_33ED250(__int64 a1, __int64 a2, __int64 a3)
{
  __m128i v4; // [rsp+0h] [rbp-10h] BYREF

  v4.m128i_i64[0] = a2;
  v4.m128i_i64[1] = a3;
  if ( (_WORD)a2 )
    return (__m128i *)sub_33ECD10(a2);
  else
    return sub_33E4710((_QWORD *)(a1 + 128), &v4) + 2;
}
