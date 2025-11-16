// Function: sub_7B8270
// Address: 0x7b8270
//
__int64 __fastcall sub_7B8270(unsigned int a1)
{
  unsigned __int32 v1; // ebx
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rsi
  char v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int16 v8; // ax
  int v9; // r13d
  int v10; // eax
  unsigned __int8 v11; // al
  int v12; // eax
  int v13; // eax
  int v14; // eax
  const void *v16; // r8
  __int64 v17; // rax
  int v18; // eax
  _BYTE *v19; // rdi
  _QWORD *v20; // rdx
  _QWORD *v21; // rsi
  char v22; // cl
  __int64 v23; // rax
  _QWORD *v24; // r10
  __int16 v25; // r9
  __int16 v26; // dx
  int v27; // eax
  _QWORD *v28; // r10
  __int16 v29; // r9
  __int16 v30; // ax
  __int64 v31; // rdi
  __int64 v32; // rax
  const void *v33; // [rsp+8h] [rbp-98h]
  unsigned __int64 v34; // [rsp+10h] [rbp-90h]
  int v35; // [rsp+18h] [rbp-88h]
  int v36; // [rsp+1Ch] [rbp-84h]
  int v37; // [rsp+20h] [rbp-80h]
  _QWORD *v39; // [rsp+30h] [rbp-70h]
  int v40; // [rsp+38h] [rbp-68h]
  unsigned __int8 v41; // [rsp+3Fh] [rbp-61h]
  _BYTE *v42; // [rsp+48h] [rbp-58h] BYREF
  _BYTE v43[16]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v44; // [rsp+60h] [rbp-40h]

  sub_7B8190();
  v41 = unk_4F063A8 & 7;
  if ( !qword_4F06410 )
  {
    v1 = xmmword_4F063C0.m128i_i32[0];
    goto LABEL_3;
  }
  v1 = sub_7B7F70((char *)qword_4F06410);
  if ( v1 != -1 )
  {
LABEL_3:
    v40 = v1 & 7;
    v37 = (v1 >> 3) & 1;
    goto LABEL_4;
  }
  v37 = 0;
  v1 = 17;
  v40 = 1;
LABEL_4:
  sub_7ADF70((__int64)v43, 0);
  if ( !a1 )
  {
    word_4F06418[0] = 7;
    if ( dword_4F17FA0
      || qword_4F06408 >= qword_4F06498 && qword_4F06408 < unk_4F06490 && !unk_4F06458 && !dword_4F17F78 )
    {
      v24 = &qword_4F063F0;
      v25 = qword_4F06408 - qword_4F06498;
      LODWORD(qword_4F063F0) = unk_4F0647C;
      if ( *(_DWORD *)&word_4F06480 && qword_4F06408 < qword_4F06488[*(int *)&word_4F06480 - 1] )
        v26 = sub_7AB680(qword_4F06408);
      else
        v26 = word_4F06480;
      v2 = qword_4F17F48;
      *((_WORD *)v24 + 2) = v25 + 1 - v26;
      if ( v2 )
        goto LABEL_6;
      goto LABEL_76;
    }
    if ( (_DWORD)qword_4F061D0 )
      qword_4F063F0 = qword_4F061D0;
    else
      sub_7B0EB0(qword_4F06408, (__int64)&qword_4F063F0);
  }
  v2 = qword_4F17F48;
  if ( !qword_4F17F48 )
  {
LABEL_76:
    qword_4F17F48 = sub_8237A0(128);
    v2 = qword_4F17F48;
  }
LABEL_6:
  sub_823800(v2);
  v3 = a1;
  v35 = 0;
  v39 = 0;
  v36 = 0;
LABEL_7:
  v4 = dword_4F07718;
  while ( dword_4F07718 && !a1 )
  {
    v34 = (unsigned __int64)qword_4F06460;
    v13 = sub_7B3E40(v3, v4);
    v5 = v13;
    if ( !v13 )
      goto LABEL_27;
    v4 = (__int64)&v42;
    v42 = &qword_4F06460[-v34];
    v16 = (const void *)sub_7B3EE0((unsigned __int8 *)v34, &v42);
    if ( unk_4F07714 && (v4 = (__int64)v42, v33 = v16, v27 = sub_7ABF60((__int64)v16, (__int64)v42), v16 = v33, v27) )
    {
      v5 = 1;
      qword_4F06460 = (_BYTE *)v34;
    }
    else
    {
      v17 = *(_QWORD *)(qword_4F17F48 + 16);
      if ( v17 )
      {
        v5 = 0;
        if ( v35 )
          goto LABEL_27;
        if ( (_BYTE *)(v17 - 1) != v42
          || (v4 = *(_QWORD *)(qword_4F17F48 + 32), v18 = memcmp(v16, (const void *)v4, v17 - 1), v5 = v18, v18) )
        {
          v5 = 0;
          sub_7B0EB0(v34, (__int64)dword_4F07508);
          v4 = *(_QWORD *)(qword_4F17F48 + 32);
          sub_6851F0(0x9B4u, v4);
          v35 = 1;
          goto LABEL_27;
        }
        v19 = qword_4F06460 - 1;
        qword_4F06408 = qword_4F06460 - 1;
        goto LABEL_89;
      }
      sub_8238B0(qword_4F17F48, v16, v42);
      v31 = qword_4F17F48;
      v32 = *(_QWORD *)(qword_4F17F48 + 16);
      v4 = v32 + 1;
      if ( (unsigned __int64)(v32 + 1) > *(_QWORD *)(qword_4F17F48 + 8) )
      {
        sub_823810(qword_4F17F48);
        v31 = qword_4F17F48;
        v32 = *(_QWORD *)(qword_4F17F48 + 16);
      }
      v5 = 0;
      *(_BYTE *)(*(_QWORD *)(v31 + 32) + v32) = 0;
      ++*(_QWORD *)(v31 + 16);
      qword_4F06408 = qword_4F06460 - 1;
    }
    if ( v35 )
      goto LABEL_27;
    v19 = (_BYTE *)qword_4F06408;
LABEL_89:
    v35 = dword_4F17FA0;
    if ( !dword_4F17FA0
      && ((unsigned __int64)v19 < qword_4F06498 || unk_4F06490 <= (unsigned __int64)v19 || unk_4F06458 || dword_4F17F78) )
    {
      if ( (_DWORD)qword_4F061D0 )
      {
        qword_4F063F0 = qword_4F061D0;
      }
      else
      {
        v4 = (__int64)&qword_4F063F0;
        sub_7B0EB0((unsigned __int64)v19, (__int64)&qword_4F063F0);
      }
    }
    else
    {
      v28 = &qword_4F063F0;
      v29 = (_WORD)v19 - qword_4F06498;
      v30 = word_4F06480;
      LODWORD(qword_4F063F0) = unk_4F0647C;
      if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > (unsigned __int64)v19 )
        v30 = sub_7AB680((unsigned __int64)v19);
      v35 = 0;
      *((_WORD *)v28 + 2) = v29 + 1 - v30;
    }
LABEL_27:
    xmmword_4F063C0.m128i_i32[0] = v1;
    sub_7AE360((__int64)v43);
    if ( !v39 )
      goto LABEL_46;
    unk_4D03D08 = 0;
    sub_7B8B50(v43, v4, v6, v7);
    unk_4D03D08 = 1;
    v36 = 1;
    v8 = word_4F06418[0];
    if ( word_4F06418[0] != 7 )
      goto LABEL_29;
LABEL_13:
    v3 = (__int64)qword_4F06410;
    if ( qword_4F06410 )
    {
      v1 = sub_7B7F70((char *)qword_4F06410);
      v9 = v1 & 7;
      v10 = 1;
      if ( (v1 & 8) == 0 )
        v10 = v37;
      v37 = v10;
      v11 = unk_4F063A8 & 7;
      if ( (unk_4F063A8 & 7) == v41 || !unk_4F063AD )
        goto LABEL_22;
      if ( !unk_4F07768 )
      {
LABEL_41:
        if ( v41 )
        {
          v3 = (unsigned int)((unk_4F063A8 & 7) != 0) + 7;
        }
        else
        {
          v41 = v11;
          v3 = 7;
        }
LABEL_43:
        sub_684AC0(v3, 0x502u);
        goto LABEL_22;
      }
    }
    else
    {
      v11 = unk_4F063A8 & 7;
      if ( v41 == (unk_4F063A8 & 7) || !unk_4F063AD )
        goto LABEL_7;
      v9 = 1;
      if ( !unk_4F07768 )
        goto LABEL_41;
    }
    if ( v41 || v40 == 2 )
    {
      if ( (unk_4F063A8 & 7) == 0 && v9 != 2 )
        goto LABEL_7;
      v3 = 8;
      goto LABEL_43;
    }
    v41 = v11;
LABEL_22:
    v12 = 2;
    if ( v9 != 2 )
      v12 = v40;
    v40 = v12;
    v4 = dword_4F07718;
  }
  xmmword_4F063C0.m128i_i32[0] = v1;
  v5 = 0;
  sub_7AE360((__int64)v43);
  if ( v39 )
    v36 = 1;
  else
LABEL_46:
    v39 = v44;
  unk_4D03D08 = 0;
  sub_7B8B50(v43, v4, v6, v7);
  unk_4D03D08 = 1;
  if ( a1 && (unsigned int)sub_693EA0(word_4F06418[0]) )
    sub_6943F0(0, 17);
  v8 = word_4F06418[0];
  if ( word_4F06418[0] == 7 )
    goto LABEL_13;
LABEL_29:
  if ( v8 == 1 && (v5 & 1) != 0 )
  {
    sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
    sub_684AC0(5u, 0xA48u);
  }
  if ( v36 )
  {
    sub_7CE770(v43, v41, 0);
    v20 = (_QWORD *)*v39;
    if ( *v39 )
    {
      v21 = v39;
      do
      {
        while ( 1 )
        {
          v23 = (__int64)v20;
          v20 = (_QWORD *)*v20;
          if ( *(_BYTE *)(v23 + 26) != 3 )
            break;
          *v21 = v23;
          v21 = (_QWORD *)v23;
          if ( !v20 )
            goto LABEL_69;
        }
        v39[2] = *(_QWORD *)(v23 + 16);
        v22 = *(_BYTE *)(v23 + 26);
        if ( v22 == 2 )
        {
          *(_QWORD *)(*(_QWORD *)(v23 + 48) + 120LL) = qword_4F08550;
          qword_4F08550 = *(_QWORD *)(v23 + 48);
        }
        else if ( v22 == 8 )
        {
          *(_QWORD *)(*(_QWORD *)(v23 + 48) + 120LL) = qword_4F08550;
          *(_QWORD *)(*(_QWORD *)(v23 + 56) + 120LL) = *(_QWORD *)(v23 + 48);
          qword_4F08550 = *(_QWORD *)(v23 + 56);
        }
        *(_QWORD *)v23 = qword_4F08558;
        qword_4F08558 = v23;
      }
      while ( v20 );
    }
    else
    {
      v21 = v39;
    }
LABEL_69:
    *v21 = 0;
    v44 = v21;
  }
  sub_7BC000(v43);
  sub_7B8260();
  if ( !unk_4F063AD )
    return 7;
  v14 = v40 | 0x10;
  if ( v37 )
    v14 = v40 | 0x18;
  xmmword_4F063C0.m128i_i32[0] = v14;
  if ( !dword_4F07718 || (v35 & 1) != 0 || !*(_QWORD *)(qword_4F17F48 + 16) )
    return 7;
  qword_4F06218 = (_QWORD *)sub_881010(*(void **)(qword_4F17F48 + 32));
  unk_4F06210 = xmmword_4F06380[0].m128i_i64[0];
  return 8;
}
