// Function: sub_15F3310
// Address: 0x15f3310
//
__int64 __fastcall sub_15F3310(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d

  v1 = *(_BYTE *)(a1 + 16);
  v2 = 1;
  if ( v1 != 55 )
    LOBYTE(v2) = (unsigned __int8)(v1 - 58) <= 1u;
  return v2;
}
