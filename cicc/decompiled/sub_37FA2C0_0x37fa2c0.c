// Function: sub_37FA2C0
// Address: 0x37fa2c0
//
__int64 __fastcall sub_37FA2C0(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v8; // rsi
  size_t v9; // r12
  size_t v10; // r13
  char *v11; // rdx
  char *v12; // r8
  void *v14; // rdi
  __int64 v15; // rdx

  if ( (char **)a1 != a2 )
  {
    v8 = *a2;
    if ( a2 + 3 == (char **)v8 )
    {
      v9 = (size_t)a2[1];
      v10 = *(_QWORD *)(a1 + 8);
      if ( v9 <= v10 )
      {
        if ( v9 )
          memmove(*(void **)a1, a2 + 3, (size_t)a2[1]);
        goto LABEL_9;
      }
      if ( v9 > *(_QWORD *)(a1 + 16) )
      {
        v15 = (__int64)a2[1];
        v10 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        sub_C8D290(a1, (const void *)(a1 + 24), v15, 1u, a5, a6);
        v11 = a2[1];
        v12 = *a2;
        if ( *a2 == &(*a2)[(_QWORD)v11] )
          goto LABEL_9;
      }
      else
      {
        v11 = a2[1];
        if ( v10 )
        {
          memmove(*(void **)a1, v8, v10);
          v8 = *a2;
          v11 = a2[1];
        }
        v12 = &v8[v10];
        if ( &v8[v10] == &v8[(_QWORD)v11] )
          goto LABEL_9;
      }
      memcpy((void *)(v10 + *(_QWORD *)a1), v12, (size_t)&v11[-v10]);
LABEL_9:
      *(_QWORD *)(a1 + 8) = v9;
      a2[1] = 0;
      return a1;
    }
    v14 = *(void **)a1;
    if ( v14 != (void *)(a1 + 24) )
    {
      _libc_free((unsigned __int64)v14);
      v8 = *a2;
    }
    *(_QWORD *)a1 = v8;
    *(_QWORD *)(a1 + 8) = a2[1];
    *(_QWORD *)(a1 + 16) = a2[2];
    *a2 = (char *)(a2 + 3);
    a2[2] = 0;
    a2[1] = 0;
  }
  return a1;
}
