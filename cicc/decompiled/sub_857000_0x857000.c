// Function: sub_857000
// Address: 0x857000
//
__int64 __fastcall sub_857000(unsigned __int64 a1, unsigned int *a2)
{
  _QWORD *v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned int v6; // r8d
  unsigned __int64 v8; // rdi
  int v9; // r13d
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  const char *i; // r13
  size_t v15; // rdx
  __int64 v16; // rsi
  int v17; // [rsp+4h] [rbp-3Ch]
  __int16 v18; // [rsp+8h] [rbp-38h]
  unsigned __int16 v19; // [rsp+Ah] [rbp-36h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]

  sub_7BC390();
  if ( *qword_4F06460 == 60 )
  {
    v2 = &qword_4F06498;
    if ( (unsigned __int64)qword_4F06460 >= qword_4F06498 )
    {
      v2 = &qword_4F06490;
      if ( (unsigned __int64)qword_4F06460 < qword_4F06490 )
        unk_4D03CFC = 1;
    }
  }
  unk_4D03D00 = 1;
  dword_4D03D1C = 1;
  sub_7B8B50(a1, a2, (__int64)v2, v3, v4, v5);
  unk_4D03CFC = 0;
  unk_4D03D00 = 0;
  if ( word_4F06418[0] == 43 )
  {
    v8 = (unsigned __int64)"<";
    v20 = dword_4F063F8;
    v19 = word_4F063FC[0];
    v17 = qword_4F063F0;
    v18 = WORD2(qword_4F063F0);
    qword_4F06C40 = 0;
    sub_7295A0("<");
    while ( 1 )
    {
      sub_7BC390();
      v9 = dword_4F063EC;
      if ( (unsigned __int16)sub_7B8B50(v8, a2, v10, v11, v12, v13) == 44 )
        break;
      if ( word_4F06418[0] == 10 )
        goto LABEL_18;
      if ( v9 )
      {
        v8 = 32;
        sub_729660(32);
      }
      for ( i = qword_4F06410; qword_4F06408 >= (unsigned __int64)i; ++i )
      {
        v8 = (unsigned int)*i;
        sub_729660(v8);
      }
    }
    sub_7295A0(">");
    v15 = qword_4F06C40;
    if ( qword_4F06C40 == 2 )
    {
LABEL_18:
      v6 = 0;
      v15 = 0;
      word_4F06418[0] = 0;
      v16 = -1;
      qword_4F06C40 = 0;
      goto LABEL_19;
    }
    v16 = qword_4F06C40 - 1LL;
    v6 = 1;
    word_4F06418[0] = 11;
LABEL_19:
    qword_4F06400 = v15;
    qword_4F06410 = (const char *)qword_4F06C50;
    qword_4F06408 = (char *)qword_4F06C50 + v16;
    dword_4F063F8 = v20;
    word_4F063FC[0] = v19;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    LODWORD(qword_4F063F0) = v17;
    WORD2(qword_4F063F0) = v18;
  }
  else
  {
    v6 = 0;
    if ( word_4F06418[0] == 11 )
    {
      v6 = 1;
      if ( qword_4F06400 == 2 )
      {
        v6 = 0;
        word_4F06418[0] = 0;
      }
    }
  }
  return v6;
}
