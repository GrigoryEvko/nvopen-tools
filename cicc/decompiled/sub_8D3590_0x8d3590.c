// Function: sub_8D3590
// Address: 0x8d3590
//
__int64 __fastcall sub_8D3590(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  __int64 v4; // rax
  char i; // dl
  unsigned __int8 v6; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v2 = 0;
  if ( v1 == 8 )
  {
    v4 = *(_QWORD *)(a1 + 160);
    for ( i = *(_BYTE *)(v4 + 140); i == 12; i = *(_BYTE *)(v4 + 140) )
      v4 = *(_QWORD *)(v4 + 160);
    v2 = 0;
    if ( i == 2 )
    {
      v6 = *(_BYTE *)(v4 + 161);
      v2 = unk_4D04000;
      if ( unk_4D04000 || (v6 & 8) == 0 )
        return v6 >> 7;
    }
  }
  return v2;
}
