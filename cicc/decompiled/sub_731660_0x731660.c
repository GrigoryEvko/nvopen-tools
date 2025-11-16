// Function: sub_731660
// Address: 0x731660
//
_BOOL8 __fastcall sub_731660(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  char i; // dl
  char v4; // al
  _BOOL8 result; // rax
  char v6; // al
  int v7; // r8d

  if ( (*(_BYTE *)(a1 + 25) & 3) != 0 )
    return 0;
  v1 = a1;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v1 + 27) & 8) != 0 )
      return 1;
    if ( !(unsigned int)sub_731540(v1) )
    {
      if ( !*(_QWORD *)(v1 + 8) )
        return 0;
      v6 = *(_BYTE *)(v1 + 24);
      if ( v6 != 1 || *(_BYTE *)(v1 + 56) != 5 )
        return v6 == 2;
      return 1;
    }
    v2 = *(_QWORD *)v1;
    for ( i = *(_BYTE *)(*(_QWORD *)v1 + 140LL); i == 12; i = *(_BYTE *)(v2 + 140) )
      v2 = *(_QWORD *)(v2 + 160);
    if ( i == 1 )
      return 0;
    if ( *(_BYTE *)(v1 + 24) != 1 )
      return 1;
    v4 = *(_BYTE *)(v1 + 56);
    if ( v4 != 91 )
      break;
    v1 = *(_QWORD *)(*(_QWORD *)(v1 + 72) + 16LL);
    if ( (*(_BYTE *)(v1 + 25) & 3) != 0 )
      return 0;
  }
  if ( v4 != 103 )
  {
    if ( (unsigned __int8)(v4 - 105) <= 4u )
      return *(_QWORD *)(v1 + 8) != 0;
    return 1;
  }
  v7 = sub_731660(*(_QWORD *)(*(_QWORD *)(v1 + 72) + 16LL));
  result = 1;
  if ( !v7 )
    return (unsigned int)sub_731660(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v1 + 72) + 16LL) + 16LL)) != 0;
  return result;
}
