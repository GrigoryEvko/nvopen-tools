// Function: sub_2D583C0
// Address: 0x2d583c0
//
__int64 __fastcall sub_2D583C0(__int64 a1)
{
  unsigned int v1; // r8d

  sub_BB5290(a1);
  v1 = 0;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 2 )
    LOBYTE(v1) = **(_BYTE **)(a1 - 32) == 17;
  return v1;
}
