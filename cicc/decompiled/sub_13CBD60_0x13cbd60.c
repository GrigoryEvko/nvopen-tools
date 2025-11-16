// Function: sub_13CBD60
// Address: 0x13cbd60
//
char __fastcall sub_13CBD60(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned __int8 v2; // dl

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u )
  {
    if ( v1 <= 3u )
    {
      v2 = *(_BYTE *)(a1 + 32);
      if ( (v2 & 0xFu) - 7 <= 1 || (((v2 & 0x30) - 16) & 0xE0) == 0 || v2 >> 6 == 2 )
        return (*(_BYTE *)(a1 + 33) & 0x1C) == 0;
      return 0;
    }
    if ( v1 == 17 )
      return sub_15E0450(a1);
    return 0;
  }
  if ( v1 != 53 || !*(_QWORD *)(a1 + 40) )
    return 0;
  if ( sub_15F2060(a1) )
    return sub_15F8F00(a1);
  else
    return 0;
}
