// Function: sub_11C55E0
// Address: 0x11c55e0
//
__int64 __fastcall sub_11C55E0(__int64 a1)
{
  int v1; // eax

  v1 = sub_B2DC70(a1);
  if ( (v1 & 0xFFFFFF55) == 0 )
    return 0;
  sub_B2DC90(a1, v1 & 0xAA);
  return 1;
}
