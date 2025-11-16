// Function: sub_7FE940
// Address: 0x7fe940
//
__m128i *__fastcall sub_7FE940(__int64 a1, int a2)
{
  __m128i *result; // rax

  sub_7FDF40(a1, 1, a2);
  result = sub_7FDF40(a1, 2, a2);
  if ( *(_BYTE *)(a1 + 174) == 2 && (*(_BYTE *)(a1 + 192) & 2) != 0 )
    return sub_7FDF40(a1, 3, a2);
  return result;
}
