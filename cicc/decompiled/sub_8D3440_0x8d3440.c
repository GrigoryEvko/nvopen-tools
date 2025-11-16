// Function: sub_8D3440
// Address: 0x8d3440
//
__int64 __fastcall sub_8D3440(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  __int64 v4; // rax
  char i; // dl

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
      v2 = unk_4D04000;
      if ( unk_4D04000 || (*(_BYTE *)(v4 + 161) & 8) == 0 )
      {
        v2 = 0;
        if ( *(_BYTE *)(v4 + 160) <= 2u )
          return (*(_DWORD *)(v4 + 160) & 0x7C800) == 0;
      }
    }
  }
  return v2;
}
