// Function: sub_8D2FF0
// Address: 0x8d2ff0
//
_BOOL8 __fastcall sub_8D2FF0(__int64 a1)
{
  while ( 1 )
  {
    if ( *(_BYTE *)(a1 + 140) != 12 )
      return (*(_BYTE *)(a1 - 8) & 0x20) != 0;
    if ( (*(_BYTE *)(a1 - 8) & 0x20) != 0 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return 1;
}
