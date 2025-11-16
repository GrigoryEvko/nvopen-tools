// Function: sub_679C10
// Address: 0x679c10
//
__int64 __fastcall sub_679C10(unsigned __int16 a1)
{
  unsigned int v1; // r13d
  char v2; // bl
  unsigned int v3; // edi
  __int64 result; // rax
  char v5; // r15
  __int16 v6; // r14
  int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int16 v11; // ax
  char v12; // [rsp+Ch] [rbp-74h]
  unsigned int v13; // [rsp+Ch] [rbp-74h]
  __int64 v14; // [rsp+10h] [rbp-70h] BYREF
  __int64 v15; // [rsp+18h] [rbp-68h]
  __int64 v16; // [rsp+20h] [rbp-60h]
  int v17; // [rsp+28h] [rbp-58h]
  _BYTE v18[4]; // [rsp+2Ch] [rbp-54h] BYREF
  int v19; // [rsp+30h] [rbp-50h]
  _BYTE v20[4]; // [rsp+34h] [rbp-4Ch] BYREF
  _BYTE v21[4]; // [rsp+38h] [rbp-48h] BYREF
  int v22; // [rsp+3Ch] [rbp-44h]
  unsigned int v23; // [rsp+40h] [rbp-40h]
  _BYTE v24[56]; // [rsp+48h] [rbp-38h] BYREF

  v1 = a1;
  v2 = a1;
  v3 = (HIBYTE(a1) ^ 1) & 1 | ((int)a1 >> 7) & 8;
  if ( (v2 & 2) != 0 )
    v3 |= 2u;
  if ( dword_4F077C4 != 2 )
    return sub_651B00(v3);
  if ( !(unsigned int)sub_651B00(v3) )
    return word_4F06418[0] == 191;
  if ( word_4F06418[0] == 187 )
  {
    sub_7ADF70(&v14, 0);
    sub_7AE360(&v14);
    sub_7B8B50(&v14, 0, v9, v10);
    v13 = sub_679C10(v1);
    sub_7BC000(&v14);
    return v13;
  }
  v5 = 0;
  if ( word_4F06418[0] == 1 && dword_4F04C64 != -1 )
  {
    v5 = (unk_4D04A18 != 0) & (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) >> 1);
    if ( v5 )
    {
      v5 = 0;
      if ( *(_BYTE *)(unk_4D04A18 + 80LL) == 3 )
      {
        v8 = *(_QWORD *)(unk_4D04A18 + 88LL);
        if ( *(_BYTE *)(v8 + 140) == 14 )
          v5 = *(_BYTE *)(v8 + 160) == 1;
      }
    }
  }
  v6 = sub_7BE840(0, 0);
  v7 = sub_651570(1, 1, 0);
  result = 1;
  switch ( word_4F06418[0] )
  {
    case 0x4Du:
      if ( dword_4F077C4 != 2 || unk_4F07778 <= 202301 )
        return result;
      break;
    case 0x107u:
      return result;
    case 0xB7u:
      goto LABEL_25;
  }
  if ( v6 != 27 && ((v2 & 0x20) == 0 || !v5) )
    return 1;
  result = 1;
  if ( v7 )
  {
    if ( word_4F06418[0] == 1 )
    {
      v12 = (unk_4D04A11 & 0x40) != 0;
      if ( !v5 )
        unk_4D04A11 |= 0x40u;
      if ( (unk_4D04A11 & 0x40) == 0 )
      {
        unk_4D04A10 &= ~0x80u;
        unk_4D04A18 = 0;
      }
LABEL_26:
      v14 = 0;
      v15 = 0x100000001LL;
      v16 = 0;
      v17 = 0;
      v19 = 1;
      v23 = dword_4F06650[0];
      v22 = 1;
      sub_7BDB60(1);
      sub_866940(1, v18, v24, v20, v21);
      sub_67A900(&v14, v1, 1);
      if ( !HIDWORD(v15) || (_DWORD)v16 )
        goto LABEL_35;
      if ( (v2 & 2) != 0 )
      {
        if ( (v2 & 1) != 0 )
          goto LABEL_33;
        if ( (v2 & 8) != 0 && unk_4D04858 )
        {
          if ( word_4F06418[0] == 75 )
            goto LABEL_35;
LABEL_33:
          if ( word_4F06418[0] == 28 )
            goto LABEL_35;
          goto LABEL_34;
        }
        if ( (v2 & 0x18) == 8 )
          goto LABEL_33;
        if ( word_4F06418[0] == 75 )
        {
LABEL_35:
          sub_679880((__int64)&v14);
          if ( word_4F06418[0] == 1 || word_4F06418[0] == 22 )
            unk_4D04A11 = unk_4D04A11 & 0xBF | (v12 << 6);
          return HIDWORD(v15);
        }
        goto LABEL_34;
      }
      if ( v2 < 0 )
      {
        if ( word_4F06418[0] == 67
          || word_4F06418[0] == 44
          || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 1) != 0 && word_4F06418[0] == 42 )
        {
          goto LABEL_35;
        }
        goto LABEL_34;
      }
      if ( ((word_4F06418[0] - 26) & 0xFFFD) != 0 && word_4F06418[0] != 74 )
      {
LABEL_34:
        HIDWORD(v15) = 0;
        goto LABEL_35;
      }
      if ( (v2 & 0x20) == 0 )
        goto LABEL_35;
      sub_7B8B50(&v14, HIDWORD(v15), (unsigned int)word_4F06418[0] - 26, (unsigned int)v16);
      if ( (unsigned int)sub_692B20(word_4F06418[0]) )
      {
        if ( word_4F06418[0] == 27 )
        {
          if ( (unsigned __int16)sub_7BE840(0, 0) == 28 )
            goto LABEL_60;
        }
        else if ( (unsigned __int16)(word_4F06418[0] - 31) <= 1u )
        {
          v11 = sub_7BE840(0, 0);
          if ( !(unsigned int)sub_692B20(v11) )
            goto LABEL_60;
        }
        if ( !v5 || (unsigned __int16)(word_4F06418[0] - 33) > 3u )
          goto LABEL_35;
      }
LABEL_60:
      if ( unk_4D0477C && word_4F06418[0] == 73 )
        goto LABEL_35;
      goto LABEL_34;
    }
LABEL_25:
    v12 = 0;
    goto LABEL_26;
  }
  return result;
}
