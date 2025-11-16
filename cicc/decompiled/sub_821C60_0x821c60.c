// Function: sub_821C60
// Address: 0x821c60
//
_BYTE **__fastcall sub_821C60(_DWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v12; // r15
  size_t v13; // rbx
  char *v14; // r14
  _DWORD *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // r15
  __int64 v21; // rbx
  _QWORD *v22; // r12
  __int64 v23; // r14
  _QWORD *v24; // r12
  char v25; // bl
  char v26; // bl
  __int64 v27; // rax
  _DWORD *v28; // [rsp+8h] [rbp-48h]
  int v29; // [rsp+10h] [rbp-40h]
  int v30; // [rsp+14h] [rbp-3Ch]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v28 = a1;
  v30 = unk_4D03D20;
  v6 = dword_4D03D1C;
  *a1 = 0;
  v29 = v6;
  qword_4F06420 = qword_4F06410;
  unk_4D03D20 = 1;
  dword_4D03D1C = 0;
  if ( (unsigned __int16)sub_7B8B50((unsigned __int64)a1, a2, a3, (__int64)&qword_4F06420, a5, a6) != 1 )
  {
    sub_6851C0(0x28u, dword_4F07508);
    dword_4D03CE0 = 1;
    goto LABEL_3;
  }
  v12 = (_QWORD *)qword_4F19408;
  v13 = qword_4F06400;
  v14 = (char *)qword_4F06410;
  if ( qword_4F19408 )
  {
    while ( 1 )
    {
      v15 = (_DWORD *)v12[1];
      a1 = v15;
      if ( v13 == strlen((const char *)v15) )
      {
        a2 = (unsigned int *)v14;
        a1 = v15;
        if ( !memcmp(v15, v14, v13) )
          break;
      }
      v12 = (_QWORD *)*v12;
      if ( !v12 )
        goto LABEL_26;
    }
    if ( (unsigned __int16)sub_7B8B50((unsigned __int64)v15, (unsigned int *)v14, v7, v8, v9, v10) != 27 )
      goto LABEL_27;
    v20 = (_QWORD *)v12[2];
LABEL_14:
    v31 = 0;
    v21 = 0;
    while ( (unsigned __int16)sub_7B8B50((unsigned __int64)a1, a2, v16, v17, v18, v19) != 10 )
    {
      if ( word_4F06418[0] == 9 )
        goto LABEL_43;
      if ( word_4F06418[0] == 28 )
      {
        if ( !v31 )
          goto LABEL_31;
        --v31;
      }
      else
      {
        v31 += word_4F06418[0] == 27;
      }
      if ( v20 )
      {
LABEL_20:
        v22 = v20;
        v23 = v21 + v20[1];
        a2 = (unsigned int *)qword_4F06410;
        a1 = (_DWORD *)v23;
        if ( (unsigned int)sub_721130(v23, (__int64)qword_4F06410, qword_4F06400)
          || *(_BYTE *)(v23 + qword_4F06400) != 32 )
        {
          while ( 1 )
          {
            v20 = (_QWORD *)*v20;
            if ( !v20 )
              break;
            a2 = (unsigned int *)v22[1];
            a1 = (_DWORD *)v20[1];
            if ( !(unsigned int)sub_721130((__int64)a1, (__int64)a2, v21) )
              goto LABEL_20;
          }
        }
        else
        {
          v21 += qword_4F06400 + 1;
        }
      }
    }
    if ( word_4F06418[0] != 28 )
    {
LABEL_43:
      sub_6851C0(0x12u, dword_4F07508);
      dword_4D03CE0 = 1;
      if ( word_4F06418[0] == 28 )
        goto LABEL_5;
      goto LABEL_4;
    }
LABEL_31:
    if ( v20 )
    {
      if ( !*(_BYTE *)(v20[1] + v21) )
      {
        if ( word_4F06418[0] == 28 )
        {
          v26 = 49;
          unk_4D03D20 = v30;
          dword_4D03D1C = v29;
          goto LABEL_41;
        }
        v25 = 1;
        qword_4F06460 = qword_4F06410;
        *v28 = 1;
        unk_4D03D20 = v30;
        dword_4D03D1C = v29;
LABEL_40:
        v26 = v25 + 48;
LABEL_41:
        v27 = sub_7AEE00(qword_4F06420, qword_4F06460 - qword_4F06420, 0, 0);
        *(_WORD *)(v27 + 52) = 768;
        qword_4F06460 = (_BYTE *)(v27 + 51);
        *(_QWORD *)(v27 + 56) = v27 + 51;
        *(_BYTE *)(v27 + 51) = v26;
        *(_QWORD *)(v27 + 64) = v27 + 52;
        *v28 = 1;
        goto LABEL_6;
      }
      v24 = v20;
      while ( 1 )
      {
        v24 = (_QWORD *)*v24;
        if ( !v24 )
          break;
        if ( !(unsigned int)sub_721130(v24[1], v20[1], v21) )
        {
          v20 = v24;
          goto LABEL_31;
        }
      }
    }
    if ( word_4F06418[0] == 28 )
    {
      v26 = 48;
      unk_4D03D20 = v30;
      dword_4D03D1C = v29;
      goto LABEL_41;
    }
    v25 = 0;
    qword_4F06460 = qword_4F06410;
    *v28 = 1;
    unk_4D03D20 = v30;
    dword_4D03D1C = v29;
    goto LABEL_40;
  }
LABEL_26:
  v20 = 0;
  if ( (unsigned __int16)sub_7B8B50((unsigned __int64)a1, a2, v7, v8, v9, v10) == 27 )
    goto LABEL_14;
LABEL_27:
  sub_6851C0(0x7Du, dword_4F07508);
  dword_4D03CE0 = 1;
LABEL_3:
  if ( word_4F06418[0] != 28 )
  {
LABEL_4:
    qword_4F06460 = qword_4F06410;
    *v28 = 1;
  }
LABEL_5:
  unk_4D03D20 = v30;
  dword_4D03D1C = v29;
LABEL_6:
  qword_4F06420 = 0;
  return &qword_4F06420;
}
