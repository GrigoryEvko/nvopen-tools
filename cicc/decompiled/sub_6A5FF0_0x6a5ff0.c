// Function: sub_6A5FF0
// Address: 0x6a5ff0
//
__int64 __fastcall sub_6A5FF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v5; // bx
  __int64 v6; // rsi
  int v7; // r10d
  __int64 v8; // rax
  __int64 v10; // rbx
  unsigned int v11; // r10d
  __int64 v12; // rax
  unsigned int v13; // r14d
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  int v19; // eax
  int v20; // eax
  __int64 i; // rax
  __int64 v22; // r14
  __int64 v23; // rax
  unsigned int v24; // [rsp+Ch] [rbp-1A4h]
  unsigned int v25; // [rsp+10h] [rbp-1A0h] BYREF
  int v26; // [rsp+14h] [rbp-19Ch] BYREF
  __int64 v27; // [rsp+18h] [rbp-198h] BYREF
  _QWORD v28[2]; // [rsp+20h] [rbp-190h] BYREF
  char v29; // [rsp+30h] [rbp-180h]
  __int64 v30; // [rsp+6Ch] [rbp-144h]

  v26 = 0;
  if ( a1 )
  {
    v5 = *(_WORD *)(a1 + 8);
    v6 = (__int64)v28;
    sub_6F8AB0(a1, (unsigned int)v28, 0, 0, (unsigned int)&v27, (unsigned int)&v25, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
    goto LABEL_32;
  }
  v13 = 0;
  v5 = word_4F06418[0];
  if ( word_4F06418[0] == 38 )
  {
    v13 = dword_4F077C0;
    if ( dword_4F077C0 )
      v13 = 512;
  }
  v27 = *(_QWORD *)&dword_4F063F8;
  v25 = dword_4F06650[0];
  sub_7B8B50(0, a2, a3, a4);
  v6 = 0;
  sub_69ED20((__int64)v28, 0, 18, v13);
  if ( dword_4F077C4 == 2 )
  {
LABEL_32:
    v6 = 1;
    if ( (unsigned int)sub_68FE10(v28, 1, 1) )
    {
      v6 = 1;
      sub_84EC30(byte_4B6D300[v5], 1, 0, 1, 0, (unsigned int)v28, 0, (__int64)&v27, v25, 0, 0, a2, 0, 0, (__int64)&v26);
    }
  }
LABEL_3:
  if ( v26 )
    goto LABEL_13;
  if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x400000FF) != 0x40000002 )
  {
LABEL_5:
    sub_6F69D0(v28, 0);
    if ( v5 != 37 )
    {
      if ( v5 > 0x25u )
      {
        if ( v5 == 38 )
        {
          if ( HIDWORD(qword_4F077B4) && dword_4F077C4 == 2 && (unsigned int)sub_8D2B80(v28[0]) )
          {
            v23 = sub_6E8E20(v28[0]);
            v11 = 30;
            v10 = v23;
          }
          else
          {
            sub_6FC8A0(v28);
            v12 = sub_6EFF80();
            v11 = 29;
            v10 = v12;
          }
LABEL_22:
          sub_6FEAC0(v11, v28, v10, a2, &v27, v25);
          if ( dword_4D04964
            && (dword_4F077C4 != 2 || !word_4D04898)
            && (unsigned int)sub_8D2A90(v10)
            && *(_BYTE *)(a2 + 16) == 2 )
          {
            sub_6E26D0(1, a2);
          }
          goto LABEL_13;
        }
      }
      else
      {
        if ( v5 == 35 )
        {
          if ( dword_4F077C4 == 2 && (unsigned int)sub_8D2E30(v28[0])
            || HIDWORD(qword_4F077B4) && (unsigned int)sub_8D2B80(v28[0]) )
          {
            v7 = 27;
          }
          else
          {
            sub_6E9580(v28);
            v7 = 27;
          }
          goto LABEL_17;
        }
        if ( v5 == 36 )
        {
LABEL_15:
          if ( !HIDWORD(qword_4F077B4) || (v19 = sub_8D2B80(v28[0]), v7 = 26, !v19) )
          {
            sub_6E9580(v28);
            v7 = 26;
          }
          goto LABEL_17;
        }
      }
      sub_721090(v28);
    }
    if ( HIDWORD(qword_4F077B4) )
    {
      v20 = sub_8D2B50(v28[0]);
      v7 = 32;
      if ( v20 )
        goto LABEL_17;
      if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_8D2B80(v28[0]) )
      {
        for ( i = v28[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v22 = *(_QWORD *)(i + 160);
        if ( !(unsigned int)sub_8D2930(v22) && !(unsigned int)sub_8D3D40(v22) )
          sub_6E6890(1695, v28);
        v7 = 28;
        goto LABEL_17;
      }
    }
    sub_6E9350(v28);
    v7 = 28;
LABEL_17:
    v24 = v7;
    sub_6FC420(v28);
    v10 = v28[0];
    v11 = v24;
    goto LABEL_22;
  }
  v14 = v28[0];
  if ( !(unsigned int)sub_7306C0(v28[0]) )
  {
    if ( v26 )
      goto LABEL_13;
    goto LABEL_5;
  }
  if ( dword_4D04800 )
  {
    if ( v5 == 36 )
    {
      v14 = v28[0];
      if ( (unsigned int)sub_8D2A90(v28[0]) )
      {
        if ( v29 == 2 )
        {
          if ( v26 )
            goto LABEL_13;
          sub_6F69D0(v28, 0);
          goto LABEL_15;
        }
      }
    }
  }
  if ( (unsigned int)sub_6E5430(v14, v6, v15, v16, v17, v18) )
    sub_6851C0(0x369u, &v27);
  sub_6E6260(a2);
  sub_6E6450(v28);
  v26 = 1;
LABEL_13:
  *(_DWORD *)(a2 + 68) = v27;
  *(_WORD *)(a2 + 72) = WORD2(v27);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  v8 = v30;
  *(_QWORD *)(a2 + 76) = v30;
  unk_4F061D8 = v8;
  return sub_6E3280(a2, &v27);
}
