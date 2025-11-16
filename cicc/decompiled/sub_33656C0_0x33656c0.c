// Function: sub_33656C0
// Address: 0x33656c0
//
__int64 __fastcall sub_33656C0(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v8; // r12
  char *v9; // rsi
  size_t v10; // r13
  size_t v11; // r15
  void *v12; // rdi
  char *v13; // rsi
  char *v14; // rdx
  __int64 v15; // rdx
  __int64 result; // rax
  void *v17; // rdi
  __int64 v18; // rdx

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
        if ( v10 && 2 * v10 )
          result = (__int64)memmove(*(void **)a1, v9, 2 * v10);
        goto LABEL_8;
      }
      if ( v10 > *(_QWORD *)(a1 + 16) )
      {
        v18 = (__int64)a2[1];
        v11 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        result = sub_C8D290(a1, (const void *)(a1 + 24), v18, 2u, a5, a6);
        v12 = *(void **)a1;
        v15 = 2LL * (_QWORD)a2[1];
        v13 = *a2;
        if ( *a2 == &(*a2)[v15] )
          goto LABEL_8;
      }
      else
      {
        v12 = *(void **)a1;
        v13 = v8;
        v14 = a2[1];
        if ( v11 )
        {
          v11 *= 2LL;
          if ( v11 )
          {
            result = (__int64)memmove(v12, v8, v11);
            v8 = *a2;
            v14 = a2[1];
            v12 = (void *)(v11 + *(_QWORD *)a1);
            v13 = &(*a2)[v11];
          }
        }
        v15 = 2LL * (_QWORD)v14;
        if ( v13 == &v8[v15] )
          goto LABEL_8;
      }
      result = (__int64)memcpy(v12, v13, v15 - v11);
LABEL_8:
      *(_QWORD *)(a1 + 8) = v10;
      a2[1] = 0;
      return result;
    }
    v17 = *(void **)a1;
    if ( v17 != (void *)(a1 + 24) )
    {
      _libc_free((unsigned __int64)v17);
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
