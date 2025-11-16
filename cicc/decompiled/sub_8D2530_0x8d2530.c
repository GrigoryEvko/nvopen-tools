// Function: sub_8D2530
// Address: 0x8d2530
//
_BOOL8 __fastcall sub_8D2530(__int64 a1)
{
  char v1; // al
  _BOOL4 v2; // r8d

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v2 = 0;
  if ( dword_4F077C4 == 2 )
  {
    if ( v1 != 7 )
    {
      v2 = v1 != 1;
      if ( v1 == 6 )
        return (*(_BYTE *)(a1 + 168) & 1) == 0;
    }
  }
  else if ( v1 != 7 )
  {
    return ((*(_BYTE *)(a1 + 141) >> 5) ^ 1) & 1;
  }
  return v2;
}
