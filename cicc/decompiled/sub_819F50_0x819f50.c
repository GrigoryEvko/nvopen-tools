// Function: sub_819F50
// Address: 0x819f50
//
__int64 __fastcall sub_819F50(unsigned __int64 a1, unsigned int *a2, _QWORD *a3, unsigned int *a4)
{
  __int64 result; // rax
  _QWORD *v5; // r13
  _BYTE *v7; // r15
  _BYTE *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  const char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // r12d
  unsigned int v16; // edx
  unsigned __int16 v17; // ax
  _BYTE *v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]

  *(_QWORD *)a2 = 0;
  *a3 = 0;
  result = word_4F06418[0];
  v20 = a1;
  if ( word_4F06418[0] != 10 )
  {
    v5 = a2;
    v19 = qword_4F194B8;
    if ( qword_4F194B8 )
    {
      v18 = qword_4F06460;
      v7 = qword_4F06460;
      qword_4F194B0 = qword_4F06460;
      while ( 1 )
      {
        a1 = (unsigned __int8)*v7;
        v8 = v7++;
        if ( !isspace(a1) )
          break;
        qword_4F06460 = v7;
      }
      a2 = a4;
      *a4 = v18 != v8;
      v12 = qword_4F06460;
      v13 = (unsigned __int8)*qword_4F06460;
      switch ( (_BYTE)v13 )
      {
        case '\'':
        case '"':
          v16 = dword_4D04954;
          goto LABEL_22;
        case '/':
          if ( qword_4F06460[1] == 42 )
          {
            qword_4F06408 = qword_4F06460;
            word_4F06418[0] = 39;
            qword_4F06410 = qword_4F06460++;
            unk_4F06400 = 1;
            v16 = dword_4D04954;
            v17 = 39;
            goto LABEL_25;
          }
          break;
        case 'L':
          v13 = (unsigned __int8)qword_4F06460[1];
          if ( (_BYTE)v13 == 39 || (_BYTE)v13 == 34 )
          {
            qword_4F06408 = qword_4F06460;
            word_4F06418[0] = 1;
            qword_4F06410 = qword_4F06460++;
            unk_4F06400 = 1;
            goto LABEL_13;
          }
          break;
      }
    }
    else
    {
      qword_4F194B0 = 0;
      if ( unk_4D0436C )
      {
        if ( *qword_4F06460 == 47 && qword_4F06460[1] == 42 && qword_4F06460[2] == 42 && qword_4F06460[3] == 47 )
        {
          a1 = (unsigned __int8)qword_4F06460[4];
          if ( !isspace(a1) )
          {
            *a4 = 0;
            word_4F06418[0] = 69;
            qword_4F06410 = qword_4F06460;
            qword_4F06408 = qword_4F06460 + 3;
            unk_4F06400 = 4;
            qword_4F06460 += 4;
            sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)&dword_4F063F8);
            sub_684B30(0x30Fu, &dword_4F063F8);
            goto LABEL_36;
          }
        }
      }
      sub_7BC390();
      *a4 = 0;
      if ( dword_4F063EC )
      {
        if ( dword_4F063EC != 1 || (v13 = dword_4D04954) == 0 )
          *a4 = 1;
      }
    }
    sub_7B8B50(a1, a2, v13, v9, v10, v11);
LABEL_36:
    v17 = word_4F06418[0];
    if ( word_4F06418[0] == 1 )
    {
LABEL_13:
      v14 = sub_819210(v20, a3);
      *v5 = v14;
      if ( !v14 )
      {
        v15 = unk_4D041B8;
        unk_4D041B8 = 0;
        if ( unk_4D04788 && unk_4F06400 == 11 && !memcmp(qword_4F06410, "__VA_ARGS__", 0xBu) )
          sub_6851C0(0x3C9u, dword_4F07508);
        unk_4D041B8 = v15;
      }
      return word_4F06418[0];
    }
    v16 = dword_4D04954;
LABEL_25:
    while ( v16 )
    {
      if ( qword_4F194B8 )
        break;
      if ( (v17 & 0xFFFD) != 5 )
        break;
      v12 = qword_4F06410;
      if ( *qword_4F06410 == 76 )
        break;
      v19 = qword_4F06408;
      qword_4F194B8 = qword_4F06408;
LABEL_22:
      qword_4F06410 = v12;
      qword_4F06408 = v12;
      word_4F06418[0] = 14;
      unk_4F06400 = 1;
      if ( v12 == (const char *)v19 )
        qword_4F194B8 = 0;
      qword_4F06460 = v12 + 1;
      v17 = 14;
    }
    return word_4F06418[0];
  }
  return result;
}
