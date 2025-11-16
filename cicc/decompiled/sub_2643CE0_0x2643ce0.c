// Function: sub_2643CE0
// Address: 0x2643ce0
//
__m128i *__fastcall sub_2643CE0(__m128i *a1, char a2)
{
  __m128i *v3; // [rsp+0h] [rbp-40h] BYREF
  __int64 v4; // [rsp+8h] [rbp-38h]
  __m128i v5[3]; // [rsp+10h] [rbp-30h] BYREF

  if ( !a2 )
  {
    sub_263F570(a1->m128i_i64, "None");
    return a1;
  }
  v5[0].m128i_i8[0] = 0;
  v3 = v5;
  v4 = 0;
  if ( (a2 & 1) != 0 )
  {
    sub_2241520((unsigned __int64 *)&v3, "NotCold");
    if ( (a2 & 2) == 0 )
      goto LABEL_4;
    goto LABEL_7;
  }
  if ( (a2 & 2) != 0 )
LABEL_7:
    sub_2241520((unsigned __int64 *)&v3, "Cold");
LABEL_4:
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v3 == v5 )
  {
    a1[1] = _mm_load_si128(v5);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v3;
    a1[1].m128i_i64[0] = v5[0].m128i_i64[0];
  }
  a1->m128i_i64[1] = v4;
  return a1;
}
