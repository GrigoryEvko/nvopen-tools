// Function: sub_8D3030
// Address: 0x8d3030
//
_BOOL8 __fastcall sub_8D3030(__int64 a1)
{
  while ( 1 )
  {
    if ( *(_BYTE *)(a1 + 140) != 12 )
      return (*(_BYTE *)(a1 - 8) & 0x40) != 0;
    if ( (*(_BYTE *)(a1 - 8) & 0x40) != 0 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  return 1;
}
