// Function: sub_39CADF0
// Address: 0x39cadf0
//
__int64 __fastcall sub_39CADF0(__int64 a1, __int64 a2, __int16 a3, __int64 a4)
{
  __int64 *v5; // r12
  unsigned __int64 *v6; // rax
  __m128i *v7; // rsi
  __int64 v9; // [rsp+0h] [rbp-40h]
  __m128i v10; // [rsp+10h] [rbp-30h] BYREF

  v5 = (__int64 *)(a2 + 8);
  if ( a4 )
  {
    v6 = *(unsigned __int64 **)(a1 + 200);
    v10.m128i_i64[0] = a4;
    v10.m128i_i64[1] = a1;
    v7 = (__m128i *)v6[77];
    if ( v7 == (__m128i *)v6[78] )
    {
      v9 = a4;
      sub_39CAC70(v6 + 76, v7, &v10);
      a4 = v9;
    }
    else
    {
      if ( v7 )
      {
        *v7 = _mm_loadu_si128(&v10);
        v7 = (__m128i *)v6[77];
      }
      v6[77] = (unsigned __int64)&v7[1];
    }
    v10.m128i_i16[2] = a3;
    v10.m128i_i16[3] = 1;
    v10.m128i_i32[0] = 4;
    v10.m128i_i64[1] = a4;
    return sub_39A31C0(v5, (__int64 *)(a1 + 88), v10.m128i_i64);
  }
  else
  {
    v10.m128i_i16[2] = a3;
    v10.m128i_i32[0] = 1;
    v10.m128i_i16[3] = 1;
    v10.m128i_i64[1] = 0;
    return sub_39A31C0(v5, (__int64 *)(a1 + 88), v10.m128i_i64);
  }
}
