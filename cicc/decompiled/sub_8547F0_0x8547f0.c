// Function: sub_8547F0
// Address: 0x8547f0
//
_QWORD *__fastcall sub_8547F0(__m128i *a1)
{
  __m128i *v2; // rbx
  unsigned __int8 v3; // di

  if ( a1 )
  {
    v2 = a1;
    do
    {
      v3 = *(_BYTE *)(v2->m128i_i64[1] + 19);
      if ( v3 != 3 )
        sub_684AA0(v3, 0x261u, (__m128i *)v2[3].m128i_i32);
      v2 = (__m128i *)v2->m128i_i64[0];
    }
    while ( v2 );
  }
  return sub_854000(a1);
}
