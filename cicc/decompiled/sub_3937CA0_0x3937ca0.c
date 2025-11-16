// Function: sub_3937CA0
// Address: 0x3937ca0
//
__m128i *__fastcall sub_3937CA0(__m128i *a1, __int64 a2, char a3)
{
  __int64 v5; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-30h] BYREF
  __m128i v7[2]; // [rsp+20h] [rbp-20h] BYREF

  if ( (unsigned __int8)sub_1C2E420(a2, "preserve_reg_abi", 0x10u, &v5) )
  {
    sub_3937240((__int64)v6, v5, a3);
    a1[2].m128i_i8[0] = 1;
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( (__m128i *)v6[0] == v7 )
    {
      a1[1] = _mm_load_si128(v7);
    }
    else
    {
      a1->m128i_i64[0] = v6[0];
      a1[1].m128i_i64[0] = v7[0].m128i_i64[0];
    }
    a1->m128i_i64[1] = v6[1];
    return a1;
  }
  else
  {
    a1[2].m128i_i8[0] = 0;
    return a1;
  }
}
