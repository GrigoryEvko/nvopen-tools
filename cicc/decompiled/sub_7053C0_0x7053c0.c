// Function: sub_7053C0
// Address: 0x7053c0
//
__int64 __fastcall sub_7053C0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v4; // rbx
  bool v5; // cf
  bool v6; // zf
  __int64 v7; // rcx
  __int64 v8; // rdi
  const char *v9; // r15
  unsigned int *v10; // rsi
  char v11; // r14
  __int64 v12; // rax
  unsigned __int16 v13; // ax
  char *v14; // rax
  char *v15; // rax
  size_t n; // [rsp+8h] [rbp-38h]

  if ( word_4F06418[0] != 55 )
    return 0;
  sub_7BDAB0(a1);
  if ( word_4F06418[0] != 7 )
  {
    if ( word_4F06418[0] != 28 && word_4F06418[0] != 55 )
    {
      v1 = 0;
      goto LABEL_18;
    }
    if ( dword_4F077C0 && qword_4F077A8 <= 0x9E33u )
      sub_6851C0(0x46Cu, &dword_4F063F8);
    return 0;
  }
  v4 = 0;
  v5 = 0;
  v6 = 1;
  v1 = 0;
  while ( 1 )
  {
    v7 = 7;
    v8 = (__int64)"memory";
    v9 = qword_4F063B8;
    v10 = (unsigned int *)qword_4F063B8;
    do
    {
      if ( !v7 )
        break;
      v5 = *(_BYTE *)v10 < *(_BYTE *)v8;
      v6 = *(_BYTE *)v10 == *(_BYTE *)v8;
      v10 = (unsigned int *)((char *)v10 + 1);
      ++v8;
      --v7;
    }
    while ( v6 );
    if ( (!v5 && !v6) == v5 )
    {
      v11 = 1;
      if ( v1 )
      {
LABEL_13:
        v12 = sub_7263C0(v8, v10);
        *(_QWORD *)v4 = v12;
        v4 = v12;
        *(_BYTE *)(v12 + 8) = v11;
        if ( v11 != 58 )
          goto LABEL_14;
LABEL_21:
        n = strlen(v9);
        v14 = (char *)sub_7247C0(n + 1);
        v15 = strncpy(v14, v9, n);
        v15[n] = 0;
        *(_QWORD *)(v4 + 16) = v15;
        goto LABEL_14;
      }
LABEL_20:
      v4 = sub_7263C0(v8, v10);
      v1 = v4;
      *(_BYTE *)(v4 + 8) = v11;
      if ( v11 != 58 )
        goto LABEL_14;
      goto LABEL_21;
    }
    if ( *qword_4F063B8 == 99 && qword_4F063B8[1] == 99 && !qword_4F063B8[2] )
    {
      v11 = 58;
      if ( (unsigned int)sub_703A60(v8, v10) )
      {
        v10 = &dword_4F063F8;
        v8 = 3622;
        sub_6851C0(0xE26u, &dword_4F063F8);
      }
LABEL_12:
      if ( v1 )
        goto LABEL_13;
      goto LABEL_20;
    }
    v8 = (__int64)qword_4F063B8;
    v11 = sub_703C10(qword_4F063B8);
    if ( v11 )
      goto LABEL_12;
    sub_6851A0(0x45Eu, &dword_4F063F8, (__int64)v9);
LABEL_14:
    sub_7BDAB0(a1);
    v13 = word_4F06418[0];
    if ( word_4F06418[0] != 67 )
      goto LABEL_15;
    sub_7BDAB0(a1);
    v5 = word_4F06418[0] < 7u;
    v6 = word_4F06418[0] == 7;
    if ( word_4F06418[0] != 7 )
    {
      sub_6851D0(0x46Eu);
      v13 = word_4F06418[0];
LABEL_15:
      v5 = v13 < 7u;
      v6 = v13 == 7;
      if ( v13 != 7 )
        break;
    }
  }
  if ( v13 != 28 && v13 != 55 )
LABEL_18:
    sub_6851D0(0x12u);
  return v1;
}
