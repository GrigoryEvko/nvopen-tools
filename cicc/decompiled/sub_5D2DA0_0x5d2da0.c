// Function: sub_5D2DA0
// Address: 0x5d2da0
//
_BOOL8 __fastcall sub_5D2DA0(__int64 a1)
{
  unsigned __int8 v1; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( v1 == 2 )
    return ((*(_BYTE *)(a1 + 161) >> 3) ^ 1) & 1;
  if ( v1 <= 2u )
    return v1 == 1;
  return (v1 & 0xEF) == 3;
}
