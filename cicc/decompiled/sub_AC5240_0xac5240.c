// Function: sub_AC5240
// Address: 0xac5240
//
bool __fastcall sub_AC5240(__int64 a1)
{
  unsigned __int8 v1; // dl
  bool result; // al

  v1 = *(_BYTE *)(a1 + 8);
  result = 1;
  if ( v1 > 3u )
  {
    result = 0;
    if ( v1 == 12 && (((*(_DWORD *)(a1 + 8) >> 8) + 16777208) & 0xFFFFFFu) <= 0x38 )
      return ((1LL << (BYTE1(*(_DWORD *)(a1 + 8)) - 8)) & 0x100000001000101LL) != 0;
  }
  return result;
}
