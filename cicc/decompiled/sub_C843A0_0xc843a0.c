// Function: sub_C843A0
// Address: 0xc843a0
//
_DWORD *__fastcall sub_C843A0(char a1, _QWORD *a2)
{
  const char **v2; // r13
  char *v3; // rax
  char *v4; // r12
  _DWORD *result; // rax
  __int64 v6; // rdi
  size_t v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD v10[4]; // [rsp+0h] [rbp-40h] BYREF
  char v11; // [rsp+20h] [rbp-20h] BYREF

  a2[1] = 0;
  if ( !a1 )
  {
    v8 = 4;
    v9 = 0;
    if ( a2[2] >= 4u )
    {
LABEL_11:
      result = (_DWORD *)(*a2 + v9);
      *result = 1886221359;
      a2[1] += 4LL;
      return result;
    }
LABEL_13:
    sub_C8D290(a2, a2 + 3, v8, 1);
    v9 = a2[1];
    goto LABEL_11;
  }
  v10[1] = "TMP";
  v2 = (const char **)v10;
  v10[2] = "TEMP";
  v10[0] = "TMPDIR";
  v10[3] = "TEMPDIR";
  v3 = getenv("TMPDIR");
  v4 = v3;
  if ( !v3 )
  {
    while ( &v11 != (char *)++v2 )
    {
      v3 = getenv(*v2);
      v4 = v3;
      if ( v3 )
        goto LABEL_5;
    }
    v9 = a2[1];
    v8 = v9 + 4;
    if ( a2[2] >= (unsigned __int64)(v9 + 4) )
      goto LABEL_11;
    goto LABEL_13;
  }
LABEL_5:
  result = (_DWORD *)strlen(v3);
  v6 = a2[1];
  v7 = (size_t)result;
  if ( (unsigned __int64)result + v6 > a2[2] )
  {
    result = (_DWORD *)sub_C8D290(a2, a2 + 3, (char *)result + v6, 1);
    v6 = a2[1];
  }
  if ( v7 )
  {
    result = memcpy((void *)(*a2 + v6), v4, v7);
    v6 = a2[1];
  }
  a2[1] = v6 + v7;
  return result;
}
