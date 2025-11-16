// Function: sub_651570
// Address: 0x651570
//
__int64 __fastcall sub_651570(int a1, int a2, unsigned int a3)
{
  int v6; // eax
  unsigned __int64 v7; // rsi
  __int64 result; // rax
  int v9; // edx
  unsigned __int8 v10; // cl
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rcx
  bool v14; // zf
  bool v15; // dl
  __int64 v16; // rcx
  int v17; // eax
  __int16 v18; // ax

  v6 = word_4F06418[0];
  if ( word_4F06418[0] == 185 )
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_54;
    goto LABEL_37;
  }
  if ( word_4F06418[0] == 183 )
  {
    v14 = (unsigned __int16)sub_7BE840(0, 0) == 25;
    v6 = word_4F06418[0];
    if ( v14 && unk_4D041A8 && dword_4F077C4 == 2 )
    {
      if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
        goto LABEL_54;
LABEL_37:
      sub_7C0F00(0, 0);
      v6 = word_4F06418[0];
    }
  }
  LOBYTE(v7) = 0;
  if ( (unsigned __int16)(v6 - 80) <= 0x30u )
    v7 = (0x1000006066221uLL >> ((unsigned __int8)v6 - 80)) & 1;
  if ( (unsigned __int16)(v6 - 331) <= 4u
    || (_WORD)v6 == 180
    || (_WORD)v6 == 165
    || (_BYTE)v7
    || (unsigned __int16)(v6 - 120) <= 2u )
  {
    goto LABEL_6;
  }
LABEL_54:
  if ( (unsigned __int16)(v6 - 126) <= 1u
    || (_WORD)v6 == 18
    || unk_4D04548 | unk_4D04558 && (unsigned __int16)(v6 - 133) <= 3u
    || (_WORD)v6 == 239
    || (unsigned __int16)(v6 - 272) <= 8u
    || (_DWORD)qword_4F077B4 && ((_WORD)v6 == 236 || (unsigned __int16)(v6 - 339) <= 0xFu) )
  {
    goto LABEL_6;
  }
  if ( (unsigned __int16)(v6 - 151) > 0x27u )
  {
    if ( (unsigned __int16)(v6 - 87) <= 0x11u )
    {
      if ( ((0x24001uLL >> ((unsigned __int8)v6 - 87)) & 1) == 0 )
        goto LABEL_13;
      goto LABEL_6;
    }
    v15 = 1;
  }
  else
  {
    v15 = ((0xC500000001uLL >> ((unsigned __int8)v6 + 105)) & 1) == 0;
  }
  if ( (_WORD)v6 != 236 && v15 )
  {
    if ( !unk_4F0775C )
      goto LABEL_13;
    v14 = (_WORD)v6 == 77;
    v6 = 77;
    if ( !v14 )
      goto LABEL_13;
  }
LABEL_6:
  if ( !a1 || !dword_4D04428 )
    return 1;
  if ( (unsigned __int16)(v6 - 80) <= 0x30u )
  {
    v13 = 0x1C70006066221LL;
    if ( _bittest64(&v13, (unsigned int)(v6 - 80)) )
      goto LABEL_12;
  }
  else if ( (_WORD)v6 == 165 || (_WORD)v6 == 180 || (unsigned __int16)(v6 - 331) <= 4u || (_WORD)v6 == 18 )
  {
    goto LABEL_12;
  }
  if ( (!(unk_4D04548 | unk_4D04558) || (unsigned __int16)(v6 - 133) > 3u)
    && (_WORD)v6 != 239
    && (unsigned __int16)(v6 - 272) > 8u
    && (!(_DWORD)qword_4F077B4 || (_WORD)v6 != 236 && (unsigned __int16)(v6 - 339) > 0xFu)
    && (dword_4F077C4 != 2 || unk_4F07778 <= 202301 || (_WORD)v6 != 77) )
  {
    return 1;
  }
LABEL_12:
  if ( (unsigned __int16)sub_7BE840(0, 0) != 73 )
    return 1;
LABEL_13:
  if ( (unsigned __int16)(word_4F06418[0] - 81) <= 0x26u )
  {
    v16 = 0x6004000001LL;
    if ( _bittest64(&v16, (unsigned int)word_4F06418[0] - 81) )
      return 1;
    LOBYTE(v9) = 0;
  }
  else
  {
    v9 = word_4F06418[0] - 164;
    LOBYTE(v9) = ((word_4F06418[0] - 164) & 0xFFFB) == 0;
    if ( (unsigned __int16)(word_4F06418[0] - 244) <= 0x16u )
      v9 |= ((unsigned __int64)&loc_790007 >> (LOBYTE(word_4F06418[0]) + 12)) & 1;
  }
  v10 = v9 | ((unsigned __int16)(word_4F06418[0] - 153) <= 1u);
  if ( v10 )
    return 1;
  v11 = a1 != 0 ? 0x4000 : 0;
  if ( (a3 & 0x10) != 0 )
    v11 = (unsigned int)v11 | 0x4000000;
  if ( a2 )
    v11 = (unsigned int)v11 | 1;
  if ( dword_4F077C4 != 2 )
  {
    if ( word_4F06418[0] != 1 )
      return 0;
LABEL_23:
    if ( (a3 & 0x10) != 0 )
      v10 = unk_4D04874 != 0;
    v12 = sub_6512E0(0, a2, 1, v10, (a3 >> 3) & 1, unk_4D04494);
    if ( unk_4D04808 && a1 )
    {
      if ( !v12 )
        goto LABEL_82;
      if ( *(_BYTE *)(v12 + 80) != 22 )
      {
        if ( (unsigned int)sub_8D3F60(*(_QWORD *)(v12 + 88)) )
        {
          v18 = sub_7BE840(0, 0);
          if ( v18 == 27 || dword_4D04428 && v18 == 73 )
            goto LABEL_82;
        }
      }
    }
    else
    {
      if ( !v12 )
        goto LABEL_82;
      if ( !a1 )
      {
LABEL_31:
        LOBYTE(result) = 1;
        return (unsigned __int8)result;
      }
    }
    if ( dword_4D04428 )
    {
      LOBYTE(result) = (unsigned __int16)sub_7BE840(0, 0) != 73;
      return (unsigned __int8)result;
    }
    goto LABEL_31;
  }
  if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
    goto LABEL_23;
  v17 = sub_7C0F00(v11, 0);
  v10 = 0;
  if ( v17 )
    goto LABEL_23;
LABEL_82:
  if ( word_4F06418[0] != 1 )
    return 0;
  if ( (unk_4D04A10 & 0x12000) == 0x12000 )
    return 1;
  if ( (unk_4D04A12 & 2) != 0
    && (unsigned __int8)(*(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 140) - 9) <= 2u
    && (*(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 177) & 0xB0) == 0x30 )
  {
    LOBYTE(result) = (unsigned __int16)sub_7BE840(0, 0) == 1;
    return (unsigned __int8)result;
  }
  return 0;
}
