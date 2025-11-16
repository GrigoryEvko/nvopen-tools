// Function: sub_73F380
// Address: 0x73f380
//
__m128i *__fastcall sub_73F380(const __m128i *a1)
{
  __m128i *v1; // r13
  const __m128i *i; // r12
  __int64 v3; // rax
  __int64 *v5; // rax

  v1 = (__m128i *)a1;
  for ( i = a1; i[8].m128i_i8[12] == 12; i = (const __m128i *)i[10].m128i_i64[0] )
    ;
  v3 = i[10].m128i_i64[1];
  do
  {
    v3 = *(_QWORD *)v3;
    if ( !v3 )
      return v1;
  }
  while ( (*(_DWORD *)(v3 + 32) & 0x3F800) == 0 );
  v1 = (__m128i *)sub_7259C0(7);
  sub_73BCD0(i, v1, 0);
  v5 = *(__int64 **)v1[10].m128i_i64[1];
  if ( !v5 )
    return v1;
  do
  {
    *((_DWORD *)v5 + 8) &= 0xFFFC07FF;
    v5 = (__int64 *)*v5;
  }
  while ( v5 );
  return v1;
}
