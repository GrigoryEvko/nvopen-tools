// Function: sub_624110
// Address: 0x624110
//
__int64 __fastcall sub_624110(__int64 a1, __int64 a2)
{
  char v3; // dl
  __int64 v4; // rax
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rsi

  v3 = *(_BYTE *)(a1 + 140);
  if ( v3 == 12 )
  {
    v4 = a1;
    do
    {
      v4 = *(_QWORD *)(v4 + 160);
      v3 = *(_BYTE *)(v4 + 140);
    }
    while ( v3 == 12 );
  }
  if ( !v3 )
    return 1;
  if ( (unsigned int)sub_8D32B0(a1) )
  {
    v6 = sub_8D46C0(a1);
    if ( v6 )
      goto LABEL_8;
    return 1;
  }
  if ( !(unsigned int)sub_8D3D10(a1) )
  {
    if ( !(unsigned int)sub_8D3D40(a1) )
    {
      if ( dword_4F077BC
        && (unk_4F04C48 != -1
         && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0
         && (*(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 6) & 2) == 0
         || (unsigned int)sub_8D4970(a1)) )
      {
        v7 = 4;
        v8 = 1715;
      }
      else
      {
        v7 = 8;
        v8 = 643;
      }
      goto LABEL_10;
    }
    return 1;
  }
  v6 = sub_8D4870(a1);
  if ( !v6 )
    return 1;
LABEL_8:
  if ( !(unsigned int)sub_8D2310(v6) )
    return 1;
  v7 = 8;
  v8 = 644;
LABEL_10:
  sub_684AA0(v7, v8, a2);
  return 0;
}
