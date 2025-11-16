// Function: sub_1099150
// Address: 0x1099150
//
__int64 __fastcall sub_1099150(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v8; // r12
  char *v9; // rsi
  size_t v10; // r13
  size_t v11; // r15
  char *v12; // rsi
  char *v13; // rdx
  void *v14; // rdi
  __int64 result; // rax
  __int64 v16; // rdx

  if ( (char **)a1 != a2 )
  {
    v8 = (char *)(a2 + 3);
    v9 = *a2;
    if ( v9 == v8 )
    {
      v10 = (size_t)a2[1];
      v11 = *(_QWORD *)(a1 + 8);
      if ( v10 <= v11 )
      {
        if ( v10 )
          result = (__int64)memmove(*(void **)a1, v9, (size_t)a2[1]);
        goto LABEL_9;
      }
      if ( v10 > *(_QWORD *)(a1 + 16) )
      {
        v16 = (__int64)a2[1];
        v11 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        result = sub_C8D290(a1, (const void *)(a1 + 24), v16, 1u, a5, a6);
        v13 = a2[1];
        v12 = *a2;
        if ( *a2 == &(*a2)[(_QWORD)v13] )
          goto LABEL_9;
      }
      else
      {
        v12 = v8;
        v13 = a2[1];
        if ( v11 )
        {
          result = (__int64)memmove(*(void **)a1, v8, v11);
          v8 = *a2;
          v13 = a2[1];
          v12 = &(*a2)[v11];
        }
        if ( v12 == &v8[(_QWORD)v13] )
          goto LABEL_9;
      }
      result = (__int64)memcpy((void *)(v11 + *(_QWORD *)a1), v12, (size_t)&v13[-v11]);
LABEL_9:
      *(_QWORD *)(a1 + 8) = v10;
      a2[1] = 0;
      return result;
    }
    v14 = *(void **)a1;
    if ( v14 != (void *)(a1 + 24) )
    {
      _libc_free(v14, v9);
      v9 = *a2;
    }
    *(_QWORD *)a1 = v9;
    *(_QWORD *)(a1 + 8) = a2[1];
    result = (__int64)a2[2];
    *(_QWORD *)(a1 + 16) = result;
    *a2 = v8;
    a2[2] = 0;
    a2[1] = 0;
  }
  return result;
}
