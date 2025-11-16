// Function: sub_E002D0
// Address: 0xe002d0
//
__int64 __fastcall sub_E002D0(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  bool v4; // al
  unsigned __int8 v5; // r10
  bool v6; // r9
  unsigned int v7; // r8d
  _BYTE *v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int16 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  char v20; // r9

  if ( !(unsigned __int8)sub_E00080(a1) )
    return 255;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    return 255;
  v2 = sub_B91C10(a2, 1);
  v3 = v2;
  if ( !v2 )
    return 255;
  v4 = sub_DFF600(v2);
  v5 = *(_BYTE *)(v3 - 16);
  v6 = (v5 & 2) != 0;
  if ( v4 )
  {
LABEL_5:
    if ( v6 )
      goto LABEL_6;
    v15 = *(_WORD *)(v3 - 16);
LABEL_23:
    v7 = (v15 >> 6) & 0xF;
    if ( v7 <= 3 )
      return 255;
    v8 = *(_BYTE **)(v3 - 8LL * ((v5 >> 2) & 0xF) - 8);
    if ( !v8 || (unsigned __int8)(*v8 - 5) > 0x1Fu )
      goto LABEL_26;
    goto LABEL_40;
  }
  if ( (*(_BYTE *)(v3 - 16) & 2) != 0 )
  {
    if ( *(_DWORD *)(v3 - 24) <= 2u )
    {
      if ( !sub_DFF600(v3) )
        return 255;
LABEL_6:
      v7 = *(_DWORD *)(v3 - 24);
      if ( v7 <= 3 )
        return 255;
      v8 = *(_BYTE **)(*(_QWORD *)(v3 - 32) + 8LL);
      if ( !v8 || (unsigned __int8)(*v8 - 5) > 0x1Fu )
      {
        v9 = 4;
LABEL_10:
        if ( v7 < (unsigned int)((_DWORD)v9 != 3) + 4 )
          return 255;
        v10 = *(_QWORD *)(v3 - 32);
        goto LABEL_29;
      }
LABEL_40:
      if ( !sub_DFF670((__int64)v8) )
      {
        v9 = 3;
        if ( v20 )
          goto LABEL_10;
LABEL_28:
        v10 = v3 - 8LL * ((v5 >> 2) & 0xF) - 16;
LABEL_29:
        v16 = *(_QWORD *)(v10 + 8 * v9);
        if ( *(_BYTE *)v16 == 1 )
        {
          v17 = *(_QWORD *)(v16 + 136);
          if ( *(_BYTE *)v17 == 17 )
          {
            v18 = *(_DWORD *)(v17 + 32) > 0x40u ? **(_QWORD **)(v17 + 24) : *(_QWORD *)(v17 + 24);
            if ( (v18 & 1) != 0 )
              return 0;
          }
        }
        return 255;
      }
      v9 = 4;
      if ( v20 )
        goto LABEL_10;
LABEL_26:
      if ( v7 <= 4 )
        return 255;
      v9 = 4;
      goto LABEL_28;
    }
    v11 = *(_QWORD *)(v3 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v3 - 16) >> 6) & 0xFu) <= 2 )
    {
      if ( !sub_DFF600(v3) )
        return 255;
      goto LABEL_23;
    }
    v11 = v3 - 8LL * ((v5 >> 2) & 0xF) - 16;
  }
  v12 = *(_QWORD *)(v11 + 16);
  if ( *(_BYTE *)v12 != 1
    || (v13 = *(_QWORD *)(v12 + 136), *(_BYTE *)v13 != 17)
    || (*(_DWORD *)(v13 + 32) > 0x40u ? (v19 = **(_QWORD **)(v13 + 24)) : (v19 = *(_QWORD *)(v13 + 24)), (v19 & 1) == 0) )
  {
    if ( !sub_DFF600(v3) )
      return 255;
    goto LABEL_5;
  }
  return 0;
}
