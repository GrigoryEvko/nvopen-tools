// Function: sub_C9FF30
// Address: 0xc9ff30
//
void __fastcall sub_C9FF30(__int64 a1, char a2)
{
  const __m128i *v2; // rbx
  __m128i **v3; // r12

  v2 = *(const __m128i **)(a1 + 64);
  if ( v2 )
  {
    v3 = (__m128i **)(a1 + 72);
    do
    {
      while ( !v2[9].m128i_i8[1] )
      {
LABEL_5:
        v2 = (const __m128i *)v2[10].m128i_i64[1];
        if ( !v2 )
          return;
      }
      if ( !v2[9].m128i_i8[0] )
      {
        sub_C9F6B0(v3, v2, (__int64)v2[5].m128i_i64, (__int64)v2[7].m128i_i64);
        if ( a2 )
          sub_C9E330((__int64)v2);
        goto LABEL_5;
      }
      sub_C9E2A0((__int64)v2);
      sub_C9F6B0(v3, v2, (__int64)v2[5].m128i_i64, (__int64)v2[7].m128i_i64);
      if ( a2 )
        sub_C9E330((__int64)v2);
      sub_C9E250((__int64)v2);
      v2 = (const __m128i *)v2[10].m128i_i64[1];
    }
    while ( v2 );
  }
}
