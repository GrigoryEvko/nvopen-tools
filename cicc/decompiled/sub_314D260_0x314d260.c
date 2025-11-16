// Function: sub_314D260
// Address: 0x314d260
//
__m128i *__fastcall sub_314D260(__m128i *a1, __int64 a2, char a3)
{
  __int64 v5; // rax
  __int64 v6; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-30h] BYREF
  __m128i v8[2]; // [rsp+20h] [rbp-20h] BYREF

  if ( (unsigned __int8)sub_CE7BB0(a2, "preserve_reg_abi", 0x10u, &v6) )
  {
    sub_314C660((__int64)v7, v6, a3);
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( (__m128i *)v7[0] == v8 )
    {
      a1[1] = _mm_load_si128(v8);
    }
    else
    {
      a1->m128i_i64[0] = v7[0];
      a1[1].m128i_i64[0] = v8[0].m128i_i64[0];
    }
    v5 = v7[1];
    a1[2].m128i_i8[0] = 1;
    a1->m128i_i64[1] = v5;
    return a1;
  }
  else
  {
    a1[2].m128i_i64[0] = 0;
    *a1 = 0;
    a1[1] = 0;
    return a1;
  }
}
