// Function: sub_73F2D0
// Address: 0x73f2d0
//
__m128i *__fastcall sub_73F2D0(const __m128i *a1)
{
  const __m128i *v1; // rax
  const __m128i *i; // r12
  __int64 **v3; // rax
  __m128i *v5; // rax
  __m128i *v6; // rbx

  v1 = a1;
  for ( i = a1; v1[8].m128i_i8[12] == 12; v1 = (const __m128i *)v1[10].m128i_i64[0] )
    ;
  v3 = (__int64 **)v1[10].m128i_i64[1];
  do
  {
    v3 = (__int64 **)*v3;
    if ( !v3 )
      return (__m128i *)a1;
  }
  while ( ((_BYTE)v3[4] & 4) == 0 );
  v5 = (__m128i *)sub_7259C0(7);
  v6 = v5;
  if ( a1[8].m128i_i8[12] == 12 )
  {
    do
      i = (const __m128i *)i[10].m128i_i64[0];
    while ( i[8].m128i_i8[12] == 12 );
  }
  sub_73BCD0(i, v5, 0);
  return v6;
}
