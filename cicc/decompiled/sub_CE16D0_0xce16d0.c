// Function: sub_CE16D0
// Address: 0xce16d0
//
bool __fastcall sub_CE16D0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rcx
  int v3; // edx

  result = 1;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( *(_BYTE *)v2
        || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80)
        || (*(_BYTE *)(v2 + 33) & 0x20) == 0
        || (result = 0, (unsigned int)(*(_DWORD *)(v2 + 36) - 68) > 3) )
      {
        result = 1;
        if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
        {
          v3 = *(_DWORD *)(v2 + 36);
          result = 0;
          if ( v3 != 11 )
            return (unsigned int)(v3 - 210) > 1;
        }
      }
    }
  }
  return result;
}
