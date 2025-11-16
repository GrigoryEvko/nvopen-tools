// Function: sub_6E1E00
// Address: 0x6e1e00
//
_DWORD *__fastcall sub_6E1E00(unsigned __int8 a1, __int64 a2, int a3, int a4)
{
  int v4; // r11d
  unsigned __int8 v5; // r10
  __int64 v6; // r9
  __int64 v8; // r13
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int8 v13; // dl
  __int16 v14; // cx
  __int64 v15; // rax
  unsigned __int8 v16; // al
  _DWORD *result; // rax
  __int64 v18; // rax

  v4 = a3;
  v5 = a1;
  v6 = a2;
  *(_BYTE *)(a2 + 16) = a1;
  v8 = qword_4D03C50;
  *(_QWORD *)a2 = qword_4D03C50;
  v9 = unk_4D03C48;
  unk_4D03C48 = 0;
  v10 = *(_DWORD *)(a2 + 17);
  *(_QWORD *)(a2 + 8) = v9;
  *(_DWORD *)(a2 + 17) = v10 & 0xFFFE0000 | 3;
  if ( !unk_4F07734 || !a1 )
  {
    *(_WORD *)(a2 + 19) &= 0xFE01u;
    v11 = dword_4F04C64;
    if ( (unsigned __int8)(a1 - 1) > 1u )
    {
      if ( dword_4F04C64 != -1 )
      {
        v12 = dword_4F04C64;
LABEL_6:
        v13 = (*(_BYTE *)(qword_4F04C68[0] + 776 * v12 + 13) & 0x20) != 0;
        goto LABEL_7;
      }
      goto LABEL_29;
    }
    goto LABEL_20;
  }
  v11 = dword_4F04C64;
  if ( dword_4F04C64 == -1 )
  {
    *(_WORD *)(a2 + 19) &= 0xFE01u;
    if ( (unsigned __int8)(a1 - 1) > 1u )
    {
LABEL_29:
      v11 = -1;
      v13 = 0;
      goto LABEL_7;
    }
LABEL_20:
    v13 = 1;
    goto LABEL_7;
  }
  v12 = dword_4F04C64;
  v13 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) >> 7;
  if ( *(char *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) >= 0 )
  {
    *(_WORD *)(a2 + 19) &= 0xFE01u;
    if ( (unsigned __int8)(v5 - 1) > 1u )
      goto LABEL_6;
    goto LABEL_20;
  }
  *(_WORD *)(a2 + 19) = *(_WORD *)(a2 + 19) & 0xFE01 | 2;
  if ( (unsigned __int8)(v5 - 1) > 1u )
    goto LABEL_6;
LABEL_7:
  v14 = *(_WORD *)(a2 + 20);
  *(_BYTE *)(a2 + 22) &= ~1u;
  *(_DWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a2 + 32) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_WORD *)(a2 + 20) = v14 & 1 | (2 * v13);
  *(_QWORD *)(a2 + 48) = 0;
  if ( (_DWORD)v11 != -1 )
    LODWORD(v11) = *(_DWORD *)(qword_4F04C68[0] + 776 * v11);
  *(_DWORD *)(a2 + 56) = v11;
  *(_QWORD *)(a2 + 64) = 0;
  v15 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a2 + 80) = 0;
  *(_QWORD *)(a2 + 88) = v15;
  *(_QWORD *)(a2 + 96) = v15;
  *(_QWORD *)(a2 + 104) = v15;
  v16 = v5;
  *(_QWORD *)(a2 + 112) = 0;
  *(_QWORD *)(a2 + 120) = 0;
  *(_QWORD *)(a2 + 128) = 0;
  *(_QWORD *)(a2 + 136) = 0;
  *(_QWORD *)(a2 + 144) = 0;
  if ( v8 )
  {
    sub_6DE9B0(1, v8, a2);
    v16 = *(_BYTE *)(v6 + 16);
  }
  qword_4D03C50 = v6;
  if ( v16 > 3u )
  {
    if ( v16 == 5 )
    {
      *(_BYTE *)(v6 + 17) = *(_BYTE *)(v6 + 17) & 0xF8 | 4;
      v18 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      *(_QWORD *)(v6 + 72) = *(_QWORD *)(v18 + 312);
      *(_QWORD *)(v6 + 80) = *(_QWORD *)(v18 + 336);
      result = &dword_4F077C4;
      if ( dword_4F077C4 != 2 )
        return result;
      goto LABEL_23;
    }
  }
  else
  {
    if ( !word_4D04898 || !v16 )
      *(_BYTE *)(v6 + 19) |= 0x40u;
    *(_WORD *)(v6 + 17) |= 0x803u;
  }
  result = &dword_4F077C4;
  if ( dword_4F077C4 != 2 )
    return result;
LABEL_23:
  if ( !v8 && !a4 )
  {
    if ( (!unk_4D044B4 || v4) && v5 == 4 )
    {
      sub_733780(0, 0, 0, 4, 0);
      result = (_DWORD *)qword_4D03C50;
      *(_QWORD *)(qword_4D03C50 + 48LL) = qword_4F06BC0;
    }
    else
    {
      result = *(_DWORD **)(qword_4F06BC0 + 24LL);
      *(_QWORD *)(v6 + 64) = result;
    }
  }
  return result;
}
