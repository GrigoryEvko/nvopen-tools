// Function: sub_73F230
// Address: 0x73f230
//
_QWORD *__fastcall sub_73F230(const __m128i *a1, __int64 a2)
{
  __m128i *j; // r12
  const __m128i *i; // rax
  __m128i *v4; // rax
  __int64 v5; // rax

  j = (__m128i *)a1;
  if ( (unsigned int)sub_8D2310(a1) )
  {
    for ( i = a1; i[8].m128i_i8[12] == 12; i = (const __m128i *)i[10].m128i_i64[0] )
      ;
    if ( a2 != *(_QWORD *)(i[10].m128i_i64[1] + 40) )
    {
      v4 = sub_73EDA0(a1, 0);
      for ( j = v4; v4[8].m128i_i8[12] == 12; v4 = (__m128i *)v4[10].m128i_i64[0] )
        ;
      v5 = v4[10].m128i_i64[1];
      *(_BYTE *)(v5 + 21) |= 1u;
      *(_QWORD *)(v5 + 40) = a2;
    }
  }
  return sub_73F0A0(j, a2);
}
