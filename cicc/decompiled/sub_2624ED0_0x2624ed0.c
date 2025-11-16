// Function: sub_2624ED0
// Address: 0x2624ed0
//
__int64 __fastcall sub_2624ED0(__int64 a1)
{
  char v1; // dl
  unsigned int v2; // r8d
  unsigned int v3; // eax

  v1 = *(_BYTE *)(a1 + 32);
  v2 = 1;
  v3 = (v1 & 0xF) - 7;
  if ( v3 > 1 )
  {
    LOBYTE(v2) = (v1 & 0x30) != 0;
    LOBYTE(v3) = (*(_BYTE *)(a1 + 32) & 0xF) != 9;
    v2 &= v3;
  }
  return v2;
}
