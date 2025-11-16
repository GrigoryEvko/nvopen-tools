// Function: sub_6E9180
// Address: 0x6e9180
//
_BOOL8 __fastcall sub_6E9180(__int64 a1)
{
  char v1; // al
  __int64 v2; // rdx
  _BOOL8 result; // rax
  unsigned __int8 v4; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 24);
    if ( v1 != 1 )
      break;
    v4 = *(_BYTE *)(a1 + 56);
    if ( v4 > 6u )
    {
      if ( v4 != 92 )
        return 0;
    }
    else if ( v4 <= 4u )
    {
      return 0;
    }
    a1 = *(_QWORD *)(a1 + 72);
  }
  if ( v1 != 2 )
    return 0;
  v2 = *(_QWORD *)(a1 + 56);
  result = 0;
  if ( *(_BYTE *)(v2 + 173) == 6 && *(_BYTE *)(v2 + 176) == 1 )
    return (*(_BYTE *)(*(_QWORD *)(v2 + 184) + 156LL) & 4) != 0;
  return result;
}
