// Function: sub_736DD0
// Address: 0x736dd0
//
__int64 __fastcall sub_736DD0(__int64 a1)
{
  __int64 v1; // rbx
  char v2; // al
  char v3; // al

  v1 = a1;
  v2 = *(_BYTE *)(a1 + 140);
  if ( v2 != 12 )
  {
    if ( (unsigned __int8)(v2 - 9) > 2u )
    {
      if ( v2 != 14 )
        return 0;
      if ( (unsigned int)sub_8D3EA0(a1) )
      {
        if ( *(_BYTE *)(a1 + 140) == 12 )
          return (*(_BYTE *)(a1 + 186) & 0x30) != 0;
        return 0;
      }
    }
    else if ( (*(_BYTE *)(a1 + 177) & 0x20) == 0 )
    {
      return 0;
    }
    return 1;
  }
  do
  {
    a1 = *(_QWORD *)(a1 + 160);
    v3 = *(_BYTE *)(a1 + 140);
  }
  while ( v3 == 12 );
  if ( (*(_BYTE *)(v1 + 186) & 0x30) != 0 )
    return 1;
  if ( (unsigned __int8)(v3 - 9) <= 2u )
    return (*(_BYTE *)(a1 + 177) & 0x20) != 0;
  if ( v3 != 14 )
    return 0;
  return (unsigned int)sub_8D3EA0(a1) == 0;
}
