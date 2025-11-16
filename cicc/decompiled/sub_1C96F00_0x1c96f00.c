// Function: sub_1C96F00
// Address: 0x1c96f00
//
bool __fastcall sub_1C96F00(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 78 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v2 + 16) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 && *(_DWORD *)(v2 + 36) == 4184 )
      return *(_BYTE *)(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)) + 16LL) == 15;
  }
  return result;
}
