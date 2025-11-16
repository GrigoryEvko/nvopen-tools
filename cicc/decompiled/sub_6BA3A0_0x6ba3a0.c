// Function: sub_6BA3A0
// Address: 0x6ba3a0
//
void *__fastcall sub_6BA3A0(__int64 a1, int a2, int a3, __int64 a4)
{
  unsigned int v5; // r13d
  int *v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // edx
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  char v23; // al
  __int64 v24; // [rsp+18h] [rbp-238h] BYREF
  _BYTE v25[160]; // [rsp+20h] [rbp-230h] BYREF
  _QWORD v26[8]; // [rsp+C0h] [rbp-190h] BYREF
  int v27; // [rsp+104h] [rbp-14Ch] BYREF
  __int64 v28; // [rsp+10Ch] [rbp-144h]

  v5 = 3;
  if ( HIDWORD(qword_4F077B4) )
  {
    v5 = qword_4F077B4;
    if ( (_DWORD)qword_4F077B4 )
    {
      v5 = 3;
    }
    else if ( qword_4F077A8 >= 0x13880u )
    {
      v5 = 3;
    }
  }
  if ( (dword_4F077C0 || dword_4F077BC && qword_4F077A8 <= 0x9C3Fu) && !word_4D04898 )
  {
    sub_6BA150(0, 0, 0, 1, v5, (__int64)v26, a4, 0);
    goto LABEL_15;
  }
  sub_6E1DD0(&v24);
  sub_6E1E00(1, v25, 0, 0);
  sub_6E2170(v24);
  v8 = 0;
  sub_69ED20((__int64)v26, 0, v5, 1);
  if ( (unsigned __int8)(*(_BYTE *)(v26[0] + 140LL) - 9) <= 2u
    && (*(_BYTE *)(*(_QWORD *)(v26[0] + 168LL) + 109LL) & 0x20) != 0 )
  {
    if ( (unsigned int)sub_6E5430(v26, 0, v26[0], v9, v10, v11) )
    {
      v8 = &v27;
      sub_6851C0(0xB50u, &v27);
    }
    goto LABEL_25;
  }
  if ( word_4D04898 )
  {
    v12 = 193;
    if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
      v12 = a2 == 0 ? 193 : 2048;
    v8 = (int *)a1;
    sub_697340(v26, a1, v12, a2, a3, 1, a4);
    goto LABEL_14;
  }
  sub_6F69D0(v26, 0);
  v14 = v26;
  v8 = (int *)a4;
  sub_6F4950(v26, a4, v15, v16, v17, v18);
  v23 = *(_BYTE *)(a4 + 173);
  if ( v23 == 1 || v23 == 12 )
  {
    if ( (unsigned int)sub_8D2930(*(_QWORD *)(a4 + 128)) )
      goto LABEL_14;
    v14 = *(_QWORD **)(a4 + 128);
    if ( (unsigned int)sub_8D3D40(v14) )
      goto LABEL_14;
    v23 = *(_BYTE *)(a4 + 173);
  }
  if ( v23 )
  {
    if ( (unsigned int)sub_6E5430(v14, a4, v19, v20, v21, v22) )
    {
      v8 = &v27;
      sub_6851C0(0x9Du, &v27);
    }
LABEL_25:
    sub_72C970(a4);
  }
LABEL_14:
  sub_6E2AC0(a4);
  sub_6E2B30(a4, v8);
  sub_6E1DF0(v24);
LABEL_15:
  unk_4F061D8 = v28;
  return &unk_4F061D8;
}
