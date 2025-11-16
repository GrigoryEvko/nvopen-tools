// Function: sub_82B7B0
// Address: 0x82b7b0
//
void __fastcall sub_82B7B0(__int64 a1, unsigned int a2)
{
  _BYTE *v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // rax

  v2 = *(_BYTE **)(a1 + 24);
  v3 = *(__int64 **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624);
  if ( v3 )
  {
    v4 = *v3;
    if ( v4 )
    {
      v5 = *(_BYTE *)(v4 + 80);
      if ( v5 == 9 || v5 == 7 )
      {
        v7 = *(_QWORD *)(v4 + 88);
      }
      else
      {
        if ( v5 != 21 )
          goto LABEL_6;
        v7 = *(_QWORD *)(*(_QWORD *)(v4 + 88) + 192LL);
      }
      if ( v7 )
        *(_BYTE *)(v7 + 176) |= 0x10u;
    }
  }
LABEL_6:
  if ( (*(_BYTE *)(a1 + 51) & 0x40) != 0 )
  {
    v6 = sub_730770(a1, 0);
    sub_82B7B0(v6, a2);
  }
  if ( a2 )
  {
    *(_BYTE *)(a1 + 49) = *(_BYTE *)(a1 + 49) & 0xEE | 1;
    if ( !v2 || !*v2 || *v2 == 3 )
      return;
LABEL_15:
    sub_733B20((_QWORD *)a1);
    sub_7340D0(a1, a2, 1);
    return;
  }
  *(_BYTE *)(a1 + 49) &= ~0x10u;
  if ( v2 && *v2 == 4 )
    goto LABEL_15;
}
