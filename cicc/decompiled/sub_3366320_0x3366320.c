// Function: sub_3366320
// Address: 0x3366320
//
bool __fastcall sub_3366320(__int64 a1)
{
  bool result; // al
  _BYTE *v2; // rdx

  result = 0;
  if ( a1 )
  {
    if ( *(_DWORD *)(a1 + 24) == 13 )
    {
      v2 = *(_BYTE **)(a1 + 96);
      if ( v2 )
      {
        if ( !*v2 )
          return (v2[33] & 3) != 1;
      }
    }
  }
  return result;
}
