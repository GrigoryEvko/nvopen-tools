// Function: sub_6D4F40
// Address: 0x6d4f40
//
_DWORD *__fastcall sub_6D4F40(int a1, int a2, int a3, _DWORD *a4, _QWORD *a5, __int64 a6)
{
  __int64 v10; // rdi
  char v11; // al
  bool v12; // bl
  __int64 v13; // rsi
  _QWORD *v14; // rdi
  char v15; // al
  __int64 v16; // rax
  char i; // dl
  _BYTE *v18; // rdi
  char v20; // al
  __int64 v21; // rsi
  __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rax
  unsigned int v29; // [rsp+1Ch] [rbp-234h] BYREF
  _BYTE v30[17]; // [rsp+20h] [rbp-230h] BYREF
  char v31; // [rsp+31h] [rbp-21Fh]
  char v32; // [rsp+32h] [rbp-21Eh]
  _QWORD v33[2]; // [rsp+C0h] [rbp-190h] BYREF
  char v34; // [rsp+D0h] [rbp-180h]
  char v35; // [rsp+100h] [rbp-150h]
  int v36; // [rsp+104h] [rbp-14Ch] BYREF
  __int64 v37; // [rsp+10Ch] [rbp-144h]
  _BYTE v38[256]; // [rsp+150h] [rbp-100h] BYREF

  v10 = 4;
  v29 = 0;
  if ( qword_4D03C50 )
    v10 = *(_BYTE *)(qword_4D03C50 + 16LL) < 4u ? 1 : 4;
  sub_6E1E00(v10, v30, 0, 0);
  v11 = v32;
  v32 |= 8u;
  if ( !a1 )
  {
    v32 = v11 | 0xA;
    if ( a2 )
    {
      if ( a3 )
        v31 |= 3u;
    }
  }
  if ( dword_4F077C4 == 2 || unk_4F07778 <= 199900 )
    sub_69ED20((__int64)v33, 0, 0, 0);
  else
    sub_69ED20((__int64)v33, 0, 1, 1);
  v12 = (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x20) != 0;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D3A70(v33[0]) )
  {
    v21 = 0;
    v22 = 193;
    if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
    {
      v27 = sub_72BA30(unk_4F06A51);
      v22 = 0;
      v21 = v27;
    }
    sub_845C60(v33, v21, v22, 2048, &v29);
  }
  v13 = v29;
  if ( !v29 )
  {
    v13 = 0;
    sub_6F69D0(v33, 0);
  }
  v14 = (_QWORD *)v33[0];
  if ( !(unsigned int)sub_8D3D40(v33[0]) )
  {
    if ( !a1 || dword_4F077C4 != 2 || unk_4F07778 <= 201401 )
      goto LABEL_15;
    v16 = v33[0];
    for ( i = *(_BYTE *)(v33[0] + 140LL); i == 12; i = *(_BYTE *)(v16 + 140) )
      v16 = *(_QWORD *)(v16 + 160);
    if ( i == 3 )
    {
      v13 = (__int64)v33;
      v14 = (_QWORD *)sub_72BA30(unk_4F06A51);
      sub_6FC3F0(v14, v33, 1);
    }
    else
    {
LABEL_15:
      if ( v34 == 1 )
      {
        v13 = 0;
        sub_6F4D20(v33, 0, a1 == 0);
      }
      v14 = v33;
      sub_6E9350(v33);
    }
  }
  v15 = v34;
  if ( v34 != 1 )
  {
    *a4 = 1;
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    *a4 = 1;
LABEL_31:
    v13 = 0;
    v18 = (_BYTE *)sub_6F6F40(v33, 0);
    *a5 = v18;
    *a5 = sub_6E2700(v18);
    *a4 = 0;
    goto LABEL_32;
  }
  v14 = v33;
  if ( (unsigned int)sub_696840((__int64)v33) )
  {
    v14 = v33;
    sub_6F4B70(v33);
  }
  v15 = v34;
  *a4 = 1;
  if ( v15 == 1 )
    goto LABEL_31;
LABEL_20:
  if ( v15 != 2 )
  {
    if ( v15 )
      sub_721090(v14);
    v18 = (_BYTE *)a6;
    sub_72C970(a6);
    sub_6E2A90();
    if ( !a1 && v12 )
    {
      *a5 = sub_7305B0(a6, v13);
      *a4 = 0;
      goto LABEL_32;
    }
    goto LABEL_35;
  }
  v18 = v38;
  v13 = a6;
  sub_72A510(v38, a6);
  sub_6E2B00();
  v20 = *(_BYTE *)(a6 + 173);
  if ( v20 != 1 )
  {
    if ( v20 && v20 != 12 )
      goto LABEL_40;
    goto LABEL_35;
  }
  if ( (dword_4F077C4 != 2 || !word_4D04898) && dword_4D04964 && (v35 & 1) != 0 )
    goto LABEL_40;
  v13 = 0;
  v18 = (_BYTE *)a6;
  v23 = sub_6210B0(a6, 0);
  if ( !a1 )
  {
    if ( v23 >= 0 )
      goto LABEL_35;
    if ( (unsigned int)sub_6E5430(a6, 0, v24, v25, v26) )
    {
      v13 = (__int64)&v36;
      sub_6851C0(0x5Eu, &v36);
    }
    goto LABEL_53;
  }
  if ( v23 < 0 )
  {
    if ( (unsigned int)sub_6E5430(a6, 0, v24, v25, v26) )
    {
      v13 = (__int64)&v36;
      sub_6851C0(0x1ADu, &v36);
    }
LABEL_53:
    v18 = (_BYTE *)a6;
    sub_72C970(a6);
    goto LABEL_35;
  }
  if ( !v23 )
  {
LABEL_40:
    v18 = v33;
    *a5 = sub_6ED0D0(v33);
    *a4 = 0;
    goto LABEL_32;
  }
LABEL_35:
  if ( *a4 )
  {
    v18 = (_BYTE *)a6;
    sub_6E2AC0(a6);
  }
LABEL_32:
  sub_6E2B30(v18, v13);
  *(_QWORD *)&dword_4F061D8 = v37;
  return &dword_4F061D8;
}
