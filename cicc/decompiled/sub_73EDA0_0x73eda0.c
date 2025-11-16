// Function: sub_73EDA0
// Address: 0x73eda0
//
__m128i *__fastcall sub_73EDA0(const __m128i *a1, int a2)
{
  const __m128i *v2; // r12
  __m128i *v3; // r13
  int v5; // r15d

  v2 = a1;
  if ( (a1[8].m128i_i8[12] & 0xFB) != 8 )
  {
    v3 = (__m128i *)sub_7259C0(7);
    sub_73BCD0(a1, v3, a2);
    return v3;
  }
  v5 = sub_8D4C10(a1, dword_4F077C4 != 2);
  if ( a1[8].m128i_i8[12] == 12 )
  {
    do
      v2 = (const __m128i *)v2[10].m128i_i64[0];
    while ( v2[8].m128i_i8[12] == 12 );
  }
  v3 = (__m128i *)sub_7259C0(7);
  sub_73BCD0(v2, v3, a2);
  if ( !v5 )
    return v3;
  return sub_73C570(v3, v5);
}
