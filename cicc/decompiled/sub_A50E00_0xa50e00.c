// Function: sub_A50E00
// Address: 0xa50e00
//
char **__fastcall sub_A50E00(__int64 *a1, __int64 a2, unsigned int a3, unsigned __int8 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r12
  _BYTE *v6; // rax
  char **result; // rax
  char *v8; // r14
  size_t v9; // rax
  void *v10; // rdi
  size_t v11; // r13

  v4 = a3;
  if ( a4 != 1 )
    sub_A50D20(a1, a2, a4);
  v5 = *a1;
  v6 = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == v6 )
  {
    v5 = sub_CB6200(v5, " ", 1);
  }
  else
  {
    *v6 = 32;
    ++*(_QWORD *)(v5 + 32);
  }
  result = &off_4B91120;
  v8 = (&off_4B91120)[v4];
  if ( v8 )
  {
    v9 = strlen((&off_4B91120)[v4]);
    v10 = *(void **)(v5 + 32);
    v11 = v9;
    result = (char **)(*(_QWORD *)(v5 + 24) - (_QWORD)v10);
    if ( v11 > (unsigned __int64)result )
    {
      return (char **)sub_CB6200(v5, v8, v11);
    }
    else if ( v11 )
    {
      result = (char **)memcpy(v10, v8, v11);
      *(_QWORD *)(v5 + 32) += v11;
    }
  }
  return result;
}
