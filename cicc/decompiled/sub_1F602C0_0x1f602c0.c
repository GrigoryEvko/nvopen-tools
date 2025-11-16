// Function: sub_1F602C0
// Address: 0x1f602c0
//
bool __fastcall sub_1F602C0(__int64 a1)
{
  char v1; // dl
  bool result; // al

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 34 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) == 0 )
    {
      result = 0;
      if ( *(_BYTE *)(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)) + 16LL) != 16 )
        return result;
      return !(*(_WORD *)(a1 + 18) & 1);
    }
    result = 0;
    if ( *(_BYTE *)(**(_QWORD **)(a1 - 8) + 16LL) == 16 )
      return !(*(_WORD *)(a1 + 18) & 1);
  }
  else
  {
    result = 0;
    if ( v1 == 73 && *(_BYTE *)(*(_QWORD *)(a1 - 24) + 16LL) == 16 )
      return sub_1F5FF70(a1) == 0;
  }
  return result;
}
