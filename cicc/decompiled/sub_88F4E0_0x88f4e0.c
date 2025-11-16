// Function: sub_88F4E0
// Address: 0x88f4e0
//
__int64 __fastcall sub_88F4E0(_DWORD *a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v7; // [rsp+Ch] [rbp-24h]

  if ( *(_DWORD *)(a2 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)a1[51])
    || (unsigned int)sub_880920(a2)
    || a1[4] )
  {
    return 0;
  }
  if ( !dword_4F077BC )
  {
    if ( (*(_BYTE *)(a2 + 81) & 0x20) == 0 )
    {
LABEL_10:
      sub_6854C0(0x2F3u, (FILE *)(a3 + 8), a2);
      return 1;
    }
    return 0;
  }
  if ( *(_BYTE *)(a2 + 80) == 19 || (*(_BYTE *)(a2 + 81) & 0x20) != 0 )
    return 0;
  if ( !a1[7] || (a4 & 1) == 0 )
    goto LABEL_10;
  v7 = a1[4];
  sub_6853B0(7u, 0x94Eu, (FILE *)(a3 + 8), a2);
  return v7;
}
