// Function: sub_BD6020
// Address: 0xbd6020
//
__int64 __fastcall sub_BD6020(__int64 a1)
{
  unsigned int v2; // r8d

  if ( *(_BYTE *)a1 == 22 )
    return sub_B2D650(a1);
  v2 = 0;
  if ( *(_BYTE *)a1 == 60 )
    return *(_WORD *)(a1 + 2) >> 7;
  return v2;
}
