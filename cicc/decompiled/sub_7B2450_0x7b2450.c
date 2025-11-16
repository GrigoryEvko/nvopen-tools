// Function: sub_7B2450
// Address: 0x7b2450
//
void sub_7B2450()
{
  _QWORD *v0; // rax
  bool i; // r13
  unsigned __int8 v2; // al
  _QWORD *v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rdi
  int v7; // r14d
  FILE *v8; // rdi
  int v9; // eax
  _QWORD *v10; // r15
  char v11; // dl
  int v12; // eax
  FILE **v13; // rcx
  FILE *v14; // rax
  int v15; // esi
  _QWORD *v16; // rax
  char *v17; // r12
  char *v18; // rdi
  char v19; // dl
  __int64 v20; // rbx
  char v21; // cl
  __int64 v22; // rcx
  __int64 v23; // rsi
  _QWORD *v24; // rax
  FILE *v25; // rdi
  int v26; // [rsp+10h] [rbp-60h] BYREF
  int v27; // [rsp+14h] [rbp-5Ch] BYREF
  char *s; // [rsp+18h] [rbp-58h] BYREF
  char *v29; // [rsp+20h] [rbp-50h] BYREF
  FILE *stream; // [rsp+28h] [rbp-48h] BYREF
  __int64 v31; // [rsp+30h] [rbp-40h] BYREF
  __int64 v32[7]; // [rsp+38h] [rbp-38h] BYREF

  v0 = qword_4F064B0;
  for ( i = (qword_4F064B0[11] & 0x10) != 0; (qword_4F064B0[11] & 0x40) != 0; v0 = qword_4F064B0 )
  {
    sub_729A00(v0[7], unk_4F06468);
    sub_7B0E20();
  }
  v2 = sub_7AFE70();
  if ( v2 == 4 || v2 <= 1u )
  {
    v3 = qword_4F064B0;
    *(_BYTE *)(qword_4F064B0[12] + 8LL) |= 1u;
  }
  else
  {
    v3 = qword_4F064B0;
  }
  v4 = v3[8];
  v5 = unk_4F06468;
  dword_4F064B8[0] = (v3[11] & 8) != 0;
  sub_729A00(v4, unk_4F06468);
  v6 = qword_4F064B0[7];
  if ( qword_4F064B0[8] != v6 )
  {
    v5 = unk_4F06468;
    sub_729A00(v6, unk_4F06468);
  }
  v7 = dword_4F17FD8;
  if ( dword_4F17FD8 || *(char *)(qword_4F064B0[8] + 72LL) < 0 )
  {
    v7 = 1;
  }
  else
  {
    v5 = unk_4F06468;
    sub_729A00(qword_4F07280, unk_4F06468);
  }
  v8 = qword_4F17FC8;
  fclose(qword_4F17FC8);
  dword_4F17FC0 = 0;
  *qword_4F064B0 = 0;
  unk_4F06478 = 0;
  if ( unk_4D0493C )
    sub_7B1260();
  if ( qword_4D04908 && byte_4F17F98 )
    sub_7AFA40((__int64)v8, v5);
  dword_4F04D98 = unk_4F04D9C;
  v9 = dword_4F17FD8;
  if ( !dword_4F17FD8 || dword_4F077C4 != 1 )
  {
    sub_858B80();
    v9 = dword_4F17FD8;
  }
  v10 = qword_4F064B0;
  v11 = *((_BYTE *)qword_4F064B0 + 88);
  if ( (v11 & 0x10) != 0 && !qword_4F04D90 && (!dword_4F04D88[0] || !unk_4F076D8) && unk_4D03C98 )
  {
    v22 = unk_4D03CA0;
    v23 = unk_4B6EB18;
    if ( unk_4D03CA0 == unk_4B6EB18 )
    {
      v22 = unk_4D03CA4;
      v23 = unk_4B6EB1C;
    }
    if ( v22 == v23 )
      unk_4D03C80 = 1;
  }
  if ( (v11 & 1) != 0 )
    --unk_4F064AC;
  v12 = v9 - 1;
  dword_4F17FD8 = v12;
  if ( v12 >= 0 )
  {
    v13 = (FILE **)(qword_4F17FE0 + 112LL * v12);
    qword_4F064B0 = v13;
    v14 = *v13;
    qword_4F17FD0 = (__int64)v13;
    if ( !v14 )
    {
      *v13 = sub_7245B0((char *)v13[2], (int *)v32, &v26);
      v24 = qword_4F064B0;
      v25 = (FILE *)*qword_4F064B0;
      if ( !*qword_4F064B0 || *((_DWORD *)qword_4F064B0 + 26) != v26 )
      {
        sub_685AD0(9u, 1702, qword_4F064B0[2], (int *)v32);
        v24 = qword_4F064B0;
        v25 = (FILE *)*qword_4F064B0;
      }
      if ( fseek(v25, v24[6], 0) )
        sub_685AD0(9u, 1702, qword_4F064B0[2], (int *)v32);
      v13 = (FILE **)qword_4F064B0;
      v14 = (FILE *)*qword_4F064B0;
    }
    v15 = *((_DWORD *)v13 + 26);
    qword_4F17FC8 = v14;
    unk_4F064A8 = v15;
    sub_722830((__int64)&unk_4F17FB0, v15);
    sub_729A40(qword_4F064B0[7], unk_4F06468 + 1, *((_DWORD *)qword_4F064B0 + 10) + 1);
    if ( unk_4D0493C )
    {
      sub_7AF280(50, 1);
      if ( !qword_4D04908 )
      {
LABEL_28:
        v16 = qword_4F064B0;
        v17 = (char *)qword_4F064B0[3];
        if ( dword_4F07680[0] && unk_4F0759C && i )
        {
          if ( sub_722B80(qword_4F076E8, (unsigned __int8 *)qword_4F076B0, 0) )
            v17 = qword_4F076B0;
          v16 = qword_4F064B0;
        }
        v18 = v17;
        sub_720B20((__int64)v17, (v16[11] & 2) != 0);
        if ( dword_4F077C4 != 1 )
          unk_4D03CD0 = qword_4F064B0[9];
        if ( (_DWORD)qword_4D04914 )
        {
          if ( dword_4D04944 )
          {
            v19 = *((_BYTE *)v10 + 88);
            if ( (v19 & 1) != 0 )
            {
              v20 = v10[8];
              v21 = *(_BYTE *)(v20 + 72);
              if ( !(v19 & 4 | v21 & 0x10) && dword_4F077C4 == 2 )
              {
                if ( unk_4D0472C )
                {
                  v18 = *(char **)(v20 + 16);
                  stream = 0;
                  if ( (unsigned int)sub_7B09E0(
                                       v18,
                                       1,
                                       1,
                                       (v21 & 8) != 0,
                                       0,
                                       1,
                                       0,
                                       0,
                                       &s,
                                       &v29,
                                       (__int64 *)&stream,
                                       &v27,
                                       &v26,
                                       &v31) )
                  {
                    if ( !v27 )
                    {
                      if ( !sub_722E50(s, *(char **)(v20 + 8), 0, 0, 0)
                        || sub_7AFEF0(s, v32, 1, 1)
                        || unk_4D0472C && (int)sub_7ABED0(s) > 0 )
                      {
                        v18 = (char *)stream;
                        fclose(stream);
                      }
                      else
                      {
                        v18 = (char *)stream;
                        sub_7B1C00(
                          (__int64)stream,
                          0,
                          (__int64)v29,
                          s,
                          0,
                          (*(_BYTE *)(v20 + 72) & 8) != 0,
                          0,
                          0,
                          1,
                          v26,
                          v31,
                          v32[0]);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        sub_7B0600((__int64)v18);
        goto LABEL_35;
      }
    }
    else if ( !qword_4D04908 )
    {
      goto LABEL_28;
    }
    sub_7AF3F0(50);
    goto LABEL_28;
  }
  qword_4F064B0 = 0;
  qword_4F17FD0 = 0;
  qword_4F17FC8 = 0;
  unk_4F064A8 = 0;
  sub_722830((__int64)&unk_4F17FB0, 0);
  if ( !v7 )
    sub_720B20((__int64)qword_4F076E8, 0);
LABEL_35:
  if ( i )
    sub_7B22D0();
}
