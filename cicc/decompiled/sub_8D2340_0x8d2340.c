// Function: sub_8D2340
// Address: 0x8d2340
//
_BOOL8 __fastcall sub_8D2340(__int64 a1)
{
  char v1; // al
  __int64 v3; // rax
  char i; // dl

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( v1 != 6 || (*(_BYTE *)(a1 + 168) & 1) != 0 )
    return 0;
  v3 = *(_QWORD *)(a1 + 160);
  for ( i = *(_BYTE *)(v3 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
    v3 = *(_QWORD *)(v3 + 160);
  return i == 7;
}
