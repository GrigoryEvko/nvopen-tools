// Function: sub_82F8B0
// Address: 0x82f8b0
//
_BOOL8 __fastcall sub_82F8B0(__int64 a1)
{
  char v2; // al
  int v3; // r8d

  if ( *(_BYTE *)(a1 + 24) != 1 )
    return 0;
  v2 = *(_BYTE *)(a1 + 56);
  if ( v2 == 100 )
  {
    v3 = 0;
  }
  else
  {
    v3 = 1;
    if ( v2 != 101 )
      return 0;
  }
  return sub_827180(v3, *(__int64 **)(a1 + 72));
}
