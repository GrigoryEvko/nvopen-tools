// Function: sub_B2DD10
// Address: 0xb2dd10
//
_BOOL8 __fastcall sub_B2DD10(__int64 a1)
{
  char v1; // al

  v1 = sub_B2DC70(a1);
  return !(((v1 & 0x40) != 0) | ((v1 & 0x10) != 0) | v1 & 1 | ((v1 & 4) != 0));
}
