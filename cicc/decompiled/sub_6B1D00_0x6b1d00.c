// Function: sub_6B1D00
// Address: 0x6b1d00
//
__int64 __fastcall sub_6B1D00(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 v5; // r14
  _QWORD *v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r8
  unsigned int v9; // r11d
  __int64 v10; // rcx
  int v11; // eax
  int v12; // edx
  __int64 v13; // rax
  bool v15; // zf
  unsigned int v16; // r14d
  int v17; // eax
  unsigned int v18; // [rsp+8h] [rbp-318h]
  unsigned int v19; // [rsp+8h] [rbp-318h]
  unsigned __int8 v20; // [rsp+17h] [rbp-309h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-308h] BYREF
  int v22; // [rsp+1Ch] [rbp-304h] BYREF
  __int64 v23; // [rsp+20h] [rbp-300h] BYREF
  __int64 v24; // [rsp+28h] [rbp-2F8h] BYREF
  _BYTE v25[352]; // [rsp+30h] [rbp-2F0h] BYREF
  _QWORD v26[8]; // [rsp+190h] [rbp-190h] BYREF
  int v27; // [rsp+1D4h] [rbp-14Ch] BYREF
  __int64 v28; // [rsp+1DCh] [rbp-144h]

  v22 = 0;
  if ( a2 )
  {
    v5 = *(_WORD *)(a2 + 8);
    v6 = v25;
    sub_6F8AB0(a2, (unsigned int)v25, (unsigned int)v26, 0, (unsigned int)&v24, (unsigned int)&v21, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  else
  {
    v6 = a1;
    v5 = word_4F06418[0];
    v24 = *(_QWORD *)&dword_4F063F8;
    v21 = dword_4F06650[0];
    sub_7B8B50(a1, 0, a3, a4);
    sub_69ED20((__int64)v26, 0, 15, 0);
    if ( dword_4F077C4 != 2 )
      goto LABEL_3;
  }
  if ( (unsigned int)sub_68FE10(v6, 1, 1) || (unsigned int)sub_68FE10(v26, 0, 1) )
    sub_84EC30(
      byte_4B6D300[v5],
      0,
      0,
      1,
      0,
      (_DWORD)v6,
      (__int64)v26,
      (__int64)&v24,
      v21,
      0,
      0,
      a3,
      0,
      0,
      (__int64)&v22);
LABEL_3:
  if ( v22 )
    goto LABEL_16;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) == 2 )
  {
    sub_68BB70(v6, v26, &v24, a3, &v22);
    if ( v22 )
      goto LABEL_16;
  }
  sub_6F69D0(v6, 0);
  if ( !HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2B80(*v6) )
  {
    if ( v5 == 40 )
    {
      sub_6E9350(v6);
      sub_6F69D0(v26, 0);
      if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_8D2B80(v26[0]) )
        goto LABEL_9;
LABEL_25:
      sub_6E9350(v26);
      goto LABEL_9;
    }
    sub_6E9580(v6);
    sub_6F69D0(v26, 0);
    if ( HIDWORD(qword_4F077B4) && (unsigned int)sub_8D2B80(v26[0]) )
      goto LABEL_9;
LABEL_8:
    sub_6E9580(v26);
    goto LABEL_9;
  }
  sub_6F69D0(v26, 0);
  if ( !HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2B80(v26[0]) )
  {
    if ( v5 == 40 )
      goto LABEL_25;
    goto LABEL_8;
  }
LABEL_9:
  v9 = v5;
  if ( dword_4F077C4 == 2 || unk_4F07778 <= 199900 || (v17 = sub_6FCD00(v5, v6, v26, &v24, &v23, &v20), v9 = v5, !v17) )
  {
    v10 = HIDWORD(qword_4F077B4);
    if ( !HIDWORD(qword_4F077B4) || (v18 = v9, v11 = sub_6FD310(v9, v6, v26, &v24, &v23, &v20), v9 = v18, !v11) )
    {
      v19 = v9;
      v23 = sub_6E8B10(v6, v26, v7, v10, v8);
      v20 = sub_6E9930(v19, v23);
      sub_6FC7D0(v23, v6, v26, v20);
    }
  }
  if ( (unsigned __int16)(v5 - 39) <= 1u
    && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0
    && *((_BYTE *)v6 + 16) != 2
    && (unsigned int)sub_6E97C0(v26) )
  {
    v15 = v5 == 39;
    v16 = 39;
    if ( !v15 )
      v16 = 179;
    if ( (unsigned int)sub_6E53E0(5, v16, &v27) )
      sub_684B30(v16, &v27);
  }
  sub_7016A0(v20, (_DWORD)v6, (unsigned int)v26, v23, a3, (unsigned int)&v24, v21);
LABEL_16:
  v12 = *((_DWORD *)v6 + 17);
  *(_WORD *)(a3 + 72) = *((_WORD *)v6 + 36);
  *(_DWORD *)(a3 + 68) = v12;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a3 + 68);
  v13 = v28;
  *(_QWORD *)(a3 + 76) = v28;
  unk_4F061D8 = v13;
  return sub_6E3280(a3, &v24);
}
