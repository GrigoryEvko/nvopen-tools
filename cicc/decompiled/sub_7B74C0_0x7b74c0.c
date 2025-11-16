// Function: sub_7B74C0
// Address: 0x7b74c0
//
__int64 __fastcall sub_7B74C0(int a1)
{
  int v1; // edx
  __int16 v2; // r9
  __int16 v3; // ax
  __int64 v4; // rcx
  int v5; // eax
  int *v6; // rsi
  __m128i *v7; // rdi
  unsigned int v8; // r13d
  unsigned int v9; // r12d
  const char *v11; // rdi
  unsigned __int8 *v12; // r12
  _QWORD *v13; // r15
  void *v14; // rax
  int *v15; // [rsp-10h] [rbp-80h]
  __m128i *v16; // [rsp-8h] [rbp-78h]
  int v17; // [rsp+1Ch] [rbp-54h] BYREF
  __m128i *v18; // [rsp+20h] [rbp-50h] BYREF
  char *v19; // [rsp+28h] [rbp-48h] BYREF
  _BYTE *v20; // [rsp+30h] [rbp-40h] BYREF
  __int64 v21; // [rsp+38h] [rbp-38h] BYREF

  v1 = a1;
  v18 = 0;
  v17 = 0;
  v19 = qword_4F06460;
  if ( !dword_4F17FA0
    && ((unsigned __int64)qword_4F06460 < qword_4F06498
     || (unsigned __int64)qword_4F06460 >= unk_4F06490
     || unk_4F06458
     || dword_4F17F78) )
  {
    if ( (_DWORD)qword_4F061D0 )
    {
      v21 = qword_4F061D0;
    }
    else
    {
      sub_7B0EB0((unsigned __int64)qword_4F06460, (__int64)&v21);
      v1 = a1;
    }
  }
  else
  {
    v2 = (_WORD)qword_4F06460 - qword_4F06498;
    v3 = word_4F06480;
    LODWORD(v21) = unk_4F0647C;
    if ( *(_DWORD *)&word_4F06480 && (unsigned __int64)qword_4F06460 < qword_4F06488[*(int *)&word_4F06480 - 1] )
    {
      v3 = sub_7AB680((unsigned __int64)qword_4F06460);
      v1 = a1;
    }
    WORD2(v21) = v2 + 1 - v3;
  }
  if ( (unsigned int)(v1 - 1) > 4 )
    sub_721090();
  v4 = 2;
  if ( v1 <= 2 )
    v4 = 2LL * (v1 == 2) + 1;
  qword_4F06460 += v4;
  v5 = sub_7B6B00(&v18, 0, v1, 39, 0, -1, v19, 0);
  v6 = v15;
  v7 = v16;
  if ( v5 )
  {
    unk_4F06208 = 8;
    if ( !unk_4D03D20 )
    {
LABEL_15:
      v7 = xmmword_4F06300;
      v8 = 5;
      sub_72C970((__int64)xmmword_4F06300);
      v9 = unk_4F06208;
      v17 = unk_4F06208;
      v19 = (char *)qword_4F06410;
      goto LABEL_16;
    }
LABEL_21:
    v8 = 0;
    goto LABEL_17;
  }
  v7 = v18;
  ++qword_4F06460;
  if ( !v18 )
  {
    unk_4F06208 = 25;
    if ( !unk_4D03D20 )
      goto LABEL_15;
    goto LABEL_21;
  }
  if ( !unk_4D03D20 )
  {
    v6 = &v17;
    sub_7CD8F0(v18, &v17, &v19);
  }
  v9 = v17;
  v8 = 5;
LABEL_16:
  if ( v9 )
    goto LABEL_22;
LABEL_17:
  if ( unk_4F07718 )
  {
    v12 = qword_4F06460;
    if ( (unsigned int)sub_7B3E40(v7, v6) )
    {
      if ( !unk_4D03D20 )
      {
        v13 = (_QWORD *)xmmword_4F06380[0].m128i_i64[0];
        if ( v13 == sub_72BA30(5u) )
        {
          v17 = 2483;
          v9 = 2483;
          v11 = qword_4F06410;
          v19 = (char *)qword_4F06410;
          goto LABEL_23;
        }
        v20 = (_BYTE *)(qword_4F06460 - v12);
        v14 = (void *)sub_7B3EE0(v12, &v20);
        qword_4F06218 = (_QWORD *)sub_881010(v14);
        unk_4F06210 = xmmword_4F06380[0].m128i_i64[0];
      }
      v9 = v17;
      if ( !v17 )
      {
        v8 = 8;
        qword_4F06408 = qword_4F06460 - 1;
        return v8;
      }
    }
    else
    {
      v9 = v17;
      if ( !v17 )
        return v8;
    }
LABEL_22:
    v11 = v19;
LABEL_23:
    sub_7B0EB0((unsigned __int64)v11, (__int64)dword_4F07508);
    sub_684AC0(8u, v9);
  }
  return v8;
}
