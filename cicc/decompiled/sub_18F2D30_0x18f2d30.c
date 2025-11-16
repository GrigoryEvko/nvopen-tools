// Function: sub_18F2D30
// Address: 0x18f2d30
//
bool __fastcall sub_18F2D30(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx
  unsigned int v3; // eax

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 78 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v2 + 16) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
    {
      v3 = *(_DWORD *)(v2 + 36);
      if ( v3 > 0x86 )
        return v3 - 137 <= 1;
      else
        return v3 > 0x84;
    }
  }
  return result;
}
