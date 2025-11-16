// Function: sub_B2DCE0
// Address: 0xb2dce0
//
__int64 __fastcall sub_B2DCE0(__int64 a1)
{
  unsigned int v1; // eax

  v1 = sub_B2DC70(a1);
  return (((unsigned __int8)((v1 >> 6) | (v1 >> 4) | v1 | (v1 >> 2)) >> 1) ^ 1) & 1;
}
