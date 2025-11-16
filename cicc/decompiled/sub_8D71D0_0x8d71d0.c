// Function: sub_8D71D0
// Address: 0x8d71d0
//
__m128i *__fastcall sub_8D71D0(__int64 a1)
{
  __int64 v1; // rbx
  __m128i *v2; // rdi
  __m128i *result; // rax
  unsigned __int16 v4; // si

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v1 = *(_QWORD *)(a1 + 168);
  v2 = *(__m128i **)(v1 + 40);
  if ( (*(_BYTE *)(v1 + 18) & 0x7F) != 0 )
    v2 = sub_73C570(v2, *(_BYTE *)(v1 + 18) & 0x7F);
  result = (__m128i *)sub_8D71C0(v2);
  v4 = *(_WORD *)(v1 + 18);
  if ( (v4 & 0x3F80) != 0 )
    return sub_73C570(result, (v4 >> 7) & 0x7F);
  return result;
}
