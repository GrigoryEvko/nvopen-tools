// Function: sub_8D3880
// Address: 0x8d3880
//
__int64 __fastcall sub_8D3880(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  _BYTE *v4; // rax
  char i; // dl
  unsigned __int8 v6; // dl

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
    v4 = *(_BYTE **)(a1 + 160);
    for ( i = v4[140]; i == 12; i = v4[140] )
      v4 = (_BYTE *)*((_QWORD *)v4 + 20);
    v2 = i == 14;
    if ( i == 2 )
    {
      v2 = unk_4D04000;
      if ( unk_4D04000 || (v4[161] & 8) == 0 )
      {
        v6 = v4[160];
        if ( v6 > 2u || (v2 = 1, (v4[162] & 4) != 0) )
        {
          if ( unk_4D043A4 || (v2 = 1, v6 != byte_4F06B90[0]) )
          {
            if ( !unk_4D043A8 )
              return (*((_DWORD *)v4 + 40) & 0x3C000) != 0;
            if ( unk_4D043A0 )
              return (*((_DWORD *)v4 + 40) & 0x3C000) != 0;
            v2 = 1;
            if ( v6 != unk_4F06B80 && v6 != unk_4F06B70 )
              return (*((_DWORD *)v4 + 40) & 0x3C000) != 0;
          }
        }
      }
    }
  }
  return v2;
}
