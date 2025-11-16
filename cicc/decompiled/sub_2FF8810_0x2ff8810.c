// Function: sub_2FF8810
// Address: 0x2ff8810
//
char __fastcall sub_2FF8810(__int64 a1, unsigned int a2, unsigned int a3)
{
  char result; // al

  if ( a3 == a2 )
    return 1;
  result = a2 == 0 || a3 == 0;
  if ( result )
    return 0;
  if ( a2 - 1 <= 0x3FFFFFFE && a3 - 1 <= 0x3FFFFFFE )
    return sub_E92070(*(_QWORD *)(a1 + 16), a2, a3);
  return result;
}
