// Function: sub_BDBCD0
// Address: 0xbdbcd0
//
bool __fastcall sub_BDBCD0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx
  unsigned int v3; // eax

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      {
        v3 = *(_DWORD *)(v2 + 36);
        if ( v3 > 0x45 )
          return v3 == 71;
        else
          return v3 > 0x43;
      }
    }
  }
  return result;
}
