// Function: sub_821AA0
// Address: 0x821aa0
//
__int64 __fastcall sub_821AA0(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 result; // rax
  __int64 *v11; // rbx
  size_t v12; // r13
  const char *v13; // r15
  __int64 *v14; // r14
  __int64 *v15; // r12
  const void *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  const char *v21; // r13
  __int64 v22; // rbx
  __int64 *v23; // r14
  int v24; // r8d
  unsigned int v25[13]; // [rsp+Ch] [rbp-34h] BYREF

  v25[0] = 0;
  if ( (unsigned __int16)sub_7B8B50(a1, a2, a3, a4, a5, a6) != 1 )
  {
    sub_6851C0(0x28u, dword_4F07508);
LABEL_3:
    result = (__int64)&dword_4D03CE0;
    dword_4D03CE0 = 1;
    return result;
  }
  v11 = (__int64 *)qword_4F19408;
  v12 = qword_4F06400;
  v13 = qword_4F06410;
  if ( qword_4F19408 )
  {
    v14 = 0;
    while ( 1 )
    {
      v16 = (const void *)v11[1];
      a1 = (unsigned __int64)v16;
      if ( v12 == strlen((const char *)v16) )
      {
        a2 = (unsigned int *)v13;
        a1 = (unsigned __int64)v16;
        if ( !memcmp(v16, v13, v12) )
          break;
      }
      v15 = (__int64 *)*v11;
      v14 = v11;
      if ( !*v11 )
        goto LABEL_12;
      v11 = (__int64 *)*v11;
    }
    v15 = v11;
    v11 = v14;
  }
  else
  {
    v15 = 0;
  }
LABEL_12:
  v25[0] = 0;
  result = sub_7B8B50(a1, a2, v6, v7, v8, v9);
  if ( (_WORD)result == 10 )
  {
    if ( !v15 )
      return result;
    goto LABEL_26;
  }
  if ( word_4F06418[0] != 27 )
  {
    sub_6851C0(0x7Du, dword_4F07508);
    goto LABEL_3;
  }
  v21 = (const char *)sub_81A350(v25, a2, v17, v18, v19, v20);
  result = v25[0];
  if ( v25[0] )
    goto LABEL_3;
  if ( v15 )
  {
    if ( v21 )
    {
      v22 = v15[2];
      if ( v22 )
      {
        v23 = 0;
        while ( 1 )
        {
          v24 = strcmp(*(const char **)(v22 + 8), v21);
          result = *(_QWORD *)v22;
          if ( !v24 )
            break;
          v23 = (__int64 *)v22;
          if ( !result )
            return result;
          v22 = *(_QWORD *)v22;
        }
        if ( v23 )
          *v23 = result;
        else
          v15[2] = result;
      }
      return result;
    }
LABEL_26:
    result = *v15;
    if ( v11 )
      *v11 = result;
    else
      qword_4F19408 = *v15;
  }
  return result;
}
