// Function: sub_8D3230
// Address: 0x8d3230
//
__int64 __fastcall sub_8D3230(__int64 a1, __int64 a2)
{
  char v2; // al
  char i; // dl
  unsigned int v4; // r8d

  while ( 1 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(a2 + 140) )
    a2 = *(_QWORD *)(a2 + 160);
  v4 = 0;
  if ( v2 != 6 )
    return v4;
  if ( (*(_BYTE *)(a1 + 168) & 1) != 0 && i == 6 )
    return *(_BYTE *)(a2 + 168) & 1;
  return 0;
}
