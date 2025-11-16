// Function: sub_B2FDA0
// Address: 0xb2fda0
//
bool __fastcall sub_B2FDA0(_BYTE *a1)
{
  bool result; // al

  result = *a1 == 0 || (unsigned __int8)(*a1 - 2) <= 1u;
  if ( result )
  {
    result = 0;
    if ( (a1[7] & 0x20) != 0 )
      return sub_B91C10(a1, 21) != 0;
  }
  return result;
}
