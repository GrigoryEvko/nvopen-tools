// Function: sub_B49E50
// Address: 0xb49e50
//
_BOOL8 __fastcall sub_B49E50(__int64 a1)
{
  char v1; // al

  v1 = sub_B49D00(a1);
  return !(((v1 & 0x40) != 0) | ((v1 & 0x10) != 0) | v1 & 1 | ((v1 & 4) != 0));
}
