// Function: sub_25F03E0
// Address: 0x25f03e0
//
char __fastcall sub_25F03E0(__int64 *a1, __int64 a2)
{
  char result; // al

  result = sub_B2D610(a2, 5);
  if ( !result )
  {
    if ( ((*(_WORD *)(a2 + 2) >> 4) & 0x3FF) == 9 )
      return 1;
    else
      return sub_D84460(*a1, a2);
  }
  return result;
}
