// Function: sub_821840
// Address: 0x821840
//
int __fastcall sub_821840(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v6; // r13
  char *v7; // r14
  _QWORD *v8; // rbx
  const void *v9; // r12
  unsigned int *v10; // rsi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  const char *v21; // r13
  __int64 **v22; // r12
  _QWORD *v23; // r12
  size_t v24; // rax
  char *v25; // rax
  _DWORD v27[9]; // [rsp+Ch] [rbp-24h] BYREF

  v27[0] = 0;
  if ( (unsigned __int16)sub_7B8B50(a1, a2, a3, a4, a5, a6) != 1 )
  {
    sub_6851C0(0x28u, dword_4F07508);
    goto LABEL_13;
  }
  v6 = qword_4F06400;
  v7 = (char *)qword_4F06410;
  if ( dword_4D04788 && qword_4F06400 == 11 )
  {
    if ( !memcmp(qword_4F06410, "__VA_ARGS__", 0xBu) )
    {
      sub_6851C0(0x3C9u, dword_4F07508);
      v6 = qword_4F06400;
      v7 = (char *)qword_4F06410;
    }
  }
  else if ( unk_4D041B8 && qword_4F06400 == 10 && !memcmp(qword_4F06410, "__VA_OPT__", 0xAu) )
  {
    sub_6851C0(0xB7Bu, dword_4F07508);
    v6 = qword_4F06400;
    v7 = (char *)qword_4F06410;
  }
  v8 = (_QWORD *)qword_4F19408;
  if ( qword_4F19408 )
  {
    while ( 1 )
    {
      v9 = (const void *)v8[1];
      if ( v6 == strlen((const char *)v9) )
      {
        v10 = (unsigned int *)v7;
        v11 = (unsigned __int64)v9;
        if ( !memcmp(v9, v7, v6) )
          break;
      }
      v8 = (_QWORD *)*v8;
      if ( !v8 )
        goto LABEL_27;
    }
  }
  else
  {
LABEL_27:
    v10 = (unsigned int *)v6;
    v11 = (unsigned __int64)v7;
    v8 = sub_81A440(v7, v6);
  }
  v27[0] = 0;
  LODWORD(v16) = sub_7B8B50(v11, v10, v12, v13, v14, v15);
  if ( (_WORD)v16 != 10 )
  {
    if ( word_4F06418[0] != 27 )
    {
      sub_6851C0(0x7Du, dword_4F07508);
LABEL_13:
      v16 = (char *)&dword_4D03CE0;
      dword_4D03CE0 = 1;
      return (int)v16;
    }
    v21 = (const char *)sub_81A350(v27, v10, v17, v18, v19, v20);
    LODWORD(v16) = v27[0];
    if ( v27[0] )
      goto LABEL_13;
    if ( v21 )
    {
      v22 = (__int64 **)v8[2];
      if ( v22 )
      {
        while ( 1 )
        {
          LODWORD(v16) = strcmp((const char *)v22[1], v21);
          if ( !(_DWORD)v16 )
            break;
          v22 = (__int64 **)*v22;
          if ( !v22 )
            goto LABEL_26;
        }
      }
      else
      {
LABEL_26:
        v23 = (_QWORD *)sub_823970(16);
        *v23 = v8[2];
        v8[2] = v23;
        v24 = strlen(v21);
        v25 = (char *)sub_823970(v24 + 1);
        v16 = strcpy(v25, v21);
        v23[1] = v16;
      }
    }
  }
  return (int)v16;
}
