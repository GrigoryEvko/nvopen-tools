// Function: sub_1549590
// Address: 0x1549590
//
char **__fastcall sub_1549590(__int64 *a1, __int64 a2, int a3, unsigned __int8 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  _BYTE *v6; // rax
  char **result; // rax
  const char *v8; // r14
  size_t v9; // rax
  void *v10; // rdi
  size_t v11; // r13

  v4 = a3;
  if ( a4 != 1 )
    sub_15494B0(a1, a2, a4);
  v5 = *a1;
  v6 = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == v6 )
  {
    v5 = sub_16E7EE0(v5, " ", 1);
  }
  else
  {
    *v6 = 32;
    ++*(_QWORD *)(v5 + 24);
  }
  result = &off_4C6F320;
  v8 = (&off_4C6F320)[v4];
  if ( v8 )
  {
    v9 = strlen((&off_4C6F320)[v4]);
    v10 = *(void **)(v5 + 24);
    v11 = v9;
    result = (char **)(*(_QWORD *)(v5 + 16) - (_QWORD)v10);
    if ( v11 > (unsigned __int64)result )
    {
      return (char **)sub_16E7EE0(v5, v8, v11);
    }
    else if ( v11 )
    {
      result = (char **)memcpy(v10, v8, v11);
      *(_QWORD *)(v5 + 24) += v11;
    }
  }
  return result;
}
