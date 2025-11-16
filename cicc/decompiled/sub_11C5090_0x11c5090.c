// Function: sub_11C5090
// Address: 0x11c5090
//
__int64 __fastcall sub_11C5090(__int64 a1)
{
  int v1; // eax

  v1 = sub_B2DC70(a1);
  if ( (v1 & 0xFFFFFFF0) == 0 )
    return 0;
  sub_B2DC90(a1, v1 & 0xF);
  return 1;
}
