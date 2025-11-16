// Function: sub_2ED0150
// Address: 0x2ed0150
//
void __fastcall sub_2ED0150(__int64 a1, __int64 a2, char a3)
{
  unsigned int v4; // eax
  bool v5; // cf
  __int64 v6; // rdi
  char v7; // dl
  unsigned int v8; // eax
  __int64 v9; // rdi

  if ( a3 )
  {
    v4 = *(_DWORD *)(a2 + 232);
    v5 = *(_DWORD *)(a1 + 308) < v4;
    v6 = a1 + 144;
    if ( !v5 )
      v4 = *(_DWORD *)(v6 + 164);
    *(_DWORD *)(a2 + 232) = v4;
    sub_2ECFB30(v6, (_QWORD *)a2);
    v7 = 1;
    if ( (*(_BYTE *)(a2 + 248) & 0x20) == 0 )
      return;
LABEL_10:
    sub_2ECA130(a1, (unsigned __int64 **)a2, v7);
    return;
  }
  v8 = *(_DWORD *)(a1 + 1028);
  v9 = a1 + 864;
  if ( *(_DWORD *)(a2 + 236) >= v8 )
    v8 = *(_DWORD *)(a2 + 236);
  *(_DWORD *)(a2 + 236) = v8;
  sub_2ECFB30(v9, (_QWORD *)a2);
  if ( (*(_BYTE *)(a2 + 248) & 0x40) != 0 )
  {
    v7 = 0;
    goto LABEL_10;
  }
}
