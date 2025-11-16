// Function: sub_ED2B40
// Address: 0xed2b40
//
bool __fastcall sub_ED2B40(__int64 a1, __int64 a2)
{
  bool result; // al
  unsigned int v3; // ecx

  result = 1;
  if ( !*(_QWORD *)(a1 + 48) )
  {
    v3 = *(_DWORD *)(a2 + 284);
    if ( v3 > 8 )
      return (*(_BYTE *)(a1 + 32) & 7) == 1;
    result = ((0x124uLL >> v3) & 1) == 0;
    if ( ((0x124uLL >> v3) & 1) == 0 )
      return (*(_BYTE *)(a1 + 32) & 7) == 1;
  }
  return result;
}
