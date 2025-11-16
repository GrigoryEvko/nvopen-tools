// Function: sub_2A4B930
// Address: 0x2a4b930
//
void __fastcall sub_2A4B930(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v8; // r13
  char *v9; // rsi
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  int v12; // r14d
  __int64 v13; // r15
  char *v14; // rsi
  __int64 v15; // rdx
  void *v16; // rdi

  if ( (char **)a1 != a2 )
  {
    v8 = (char *)(a2 + 2);
    v9 = *a2;
    if ( v9 == v8 )
    {
      v10 = *((unsigned int *)a2 + 2);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *((_DWORD *)a2 + 2);
      if ( v10 <= v11 )
      {
        if ( *((_DWORD *)a2 + 2) )
          memmove(*(void **)a1, v9, 8 * v10);
        goto LABEL_9;
      }
      if ( v10 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 8u, a5, a6);
        v11 = 0;
        v15 = 8LL * *((unsigned int *)a2 + 2);
        v14 = *a2;
        if ( *a2 == &(*a2)[v15] )
          goto LABEL_9;
      }
      else
      {
        v13 = 8 * v11;
        v14 = v8;
        if ( *(_DWORD *)(a1 + 8) )
        {
          memmove(*(void **)a1, v8, 8 * v11);
          v8 = *a2;
          v10 = *((unsigned int *)a2 + 2);
          v11 = v13;
          v14 = &(*a2)[v13];
        }
        v15 = 8 * v10;
        if ( v14 == &v8[v15] )
          goto LABEL_9;
      }
      memcpy((void *)(v11 + *(_QWORD *)a1), v14, v15 - v11);
LABEL_9:
      *(_DWORD *)(a1 + 8) = v12;
      *((_DWORD *)a2 + 2) = 0;
      return;
    }
    v16 = *(void **)a1;
    if ( v16 != (void *)(a1 + 16) )
    {
      _libc_free((unsigned __int64)v16);
      v9 = *a2;
    }
    *(_QWORD *)a1 = v9;
    *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
    *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
    *a2 = v8;
    a2[1] = 0;
  }
}
