// Function: sub_9C3060
// Address: 0x9c3060
//
char *__fastcall sub_9C3060(__int64 a1, char **a2)
{
  char *v4; // r12
  char *v5; // rsi
  size_t v6; // r13
  size_t v7; // r15
  char *v8; // rsi
  char *v9; // rdx
  void *v10; // rdi
  char *result; // rax
  char *v12; // rdx

  if ( (char **)a1 != a2 )
  {
    v4 = (char *)(a2 + 3);
    v5 = *a2;
    if ( v5 == v4 )
    {
      v6 = (size_t)a2[1];
      v7 = *(_QWORD *)(a1 + 8);
      if ( v6 <= v7 )
      {
        if ( v6 )
          result = (char *)memmove(*(void **)a1, v5, (size_t)a2[1]);
        goto LABEL_9;
      }
      if ( v6 > *(_QWORD *)(a1 + 16) )
      {
        v12 = a2[1];
        v7 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        result = (char *)sub_C8D290(a1, a1 + 24, v12, 1);
        v9 = a2[1];
        v8 = *a2;
        if ( *a2 == &(*a2)[(_QWORD)v9] )
          goto LABEL_9;
      }
      else
      {
        v8 = v4;
        v9 = a2[1];
        if ( v7 )
        {
          result = (char *)memmove(*(void **)a1, v4, v7);
          v4 = *a2;
          v9 = a2[1];
          v8 = &(*a2)[v7];
        }
        if ( v8 == &v4[(_QWORD)v9] )
          goto LABEL_9;
      }
      result = (char *)memcpy((void *)(v7 + *(_QWORD *)a1), v8, (size_t)&v9[-v7]);
LABEL_9:
      *(_QWORD *)(a1 + 8) = v6;
      a2[1] = 0;
      return result;
    }
    v10 = *(void **)a1;
    if ( v10 != (void *)(a1 + 24) )
    {
      _libc_free(v10, v5);
      v5 = *a2;
    }
    *(_QWORD *)a1 = v5;
    *(_QWORD *)(a1 + 8) = a2[1];
    result = a2[2];
    *(_QWORD *)(a1 + 16) = result;
    *a2 = v4;
    a2[2] = 0;
    a2[1] = 0;
  }
  return result;
}
