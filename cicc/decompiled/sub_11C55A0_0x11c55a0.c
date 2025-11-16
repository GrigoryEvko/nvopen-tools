// Function: sub_11C55A0
// Address: 0x11c55a0
//
__int64 __fastcall sub_11C55A0(__int64 a1)
{
  int v1; // eax

  v1 = sub_B2DC70(a1);
  if ( (v1 & 0xFFFFFFAA) == 0 )
    return 0;
  sub_B2DC90(a1, v1 & 0x55);
  return 1;
}
