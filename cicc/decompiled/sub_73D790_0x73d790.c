// Function: sub_73D790
// Address: 0x73d790
//
__m128i *__fastcall sub_73D790(__int64 a1)
{
  const __m128i *v1; // r12

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v1 = *(const __m128i **)(a1 + 160);
  if ( (unsigned int)sub_8D32E0(v1) )
    return (__m128i *)sub_8D46C0(v1);
  else
    return sub_73D720(v1);
}
