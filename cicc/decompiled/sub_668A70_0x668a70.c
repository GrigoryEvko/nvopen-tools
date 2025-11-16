// Function: sub_668A70
// Address: 0x668a70
//
void __fastcall sub_668A70(
        int a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        _DWORD *a7,
        _QWORD *a8,
        _DWORD *a9)
{
  __int64 v12; // rdi
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rcx

  if ( !unk_4F0775C && (!(_DWORD)qword_4F077B4 || dword_4F077C4 != 2 || !qword_4F077A0)
    || unk_4F07758 && (*(_BYTE *)(a4 + 125) & 2) == 0 && ((*(_BYTE *)a6 & 4) != 0 || (a3 & 2) == 0) )
  {
    *(_WORD *)(a4 + 124) &= 0xFE7Fu;
    sub_668230(77, a3, a4, a5, a2, a6, a9);
    return;
  }
  if ( !a1 )
  {
    v12 = (*(_BYTE *)(a4 + 125) & 2) == 0 ? 1598 : 2542;
    goto LABEL_16;
  }
  if ( (*a6 & 0x804LL) == 0 )
  {
    if ( (a3 & 8) == 0 || (v13 = *(_BYTE *)(a4 + 125) & 2) == 0 )
    {
      *a7 = 24;
      v14 = sub_72B6D0(a4 + 104, (*(_BYTE *)(a4 + 125) & 2) != 0);
      v15 = *(_QWORD *)(a4 + 408);
      *(_QWORD *)(a4 + 304) = v14;
      *(_QWORD *)(*(_QWORD *)(v14 + 168) + 32LL) = v15;
      *a8 = *(_QWORD *)(a4 + 304);
      if ( !unk_4F0775C )
        sub_684B30(3422, a4 + 104);
      goto LABEL_17;
    }
LABEL_20:
    if ( (a3 & 8) != 0 )
    {
      v12 = 2542;
      goto LABEL_16;
    }
    goto LABEL_10;
  }
  if ( *(_BYTE *)(a4 + 125) & 2 )
    goto LABEL_20;
LABEL_10:
  v12 = 84;
LABEL_16:
  sub_6851C0(v12, a4 + 104);
  *a7 = 26;
  *a8 = sub_72C930(v12);
  *a9 = 1;
  sub_643D80(a4);
LABEL_17:
  *a6 |= 4uLL;
}
