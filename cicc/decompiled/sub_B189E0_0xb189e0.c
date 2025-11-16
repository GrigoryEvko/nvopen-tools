// Function: sub_B189E0
// Address: 0xb189e0
//
void __fastcall sub_B189E0(__int64 a1, char **a2)
{
  char *v4; // r13
  char *v5; // rsi
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rax
  int v8; // r14d
  __int64 v9; // r15
  char *v10; // rsi
  __int64 v11; // rdx
  void *v12; // rdi

  if ( (char **)a1 != a2 )
  {
    v4 = (char *)(a2 + 2);
    v5 = *a2;
    if ( v5 == v4 )
    {
      v6 = *((unsigned int *)a2 + 2);
      v7 = *(unsigned int *)(a1 + 8);
      v8 = *((_DWORD *)a2 + 2);
      if ( v6 <= v7 )
      {
        if ( *((_DWORD *)a2 + 2) )
          memmove(*(void **)a1, v5, 4 * v6);
        goto LABEL_9;
      }
      if ( v6 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        sub_C8D5F0(a1, a1 + 16, v6, 4);
        v7 = 0;
        v11 = 4LL * *((unsigned int *)a2 + 2);
        v10 = *a2;
        if ( *a2 == &(*a2)[v11] )
          goto LABEL_9;
      }
      else
      {
        v9 = 4 * v7;
        v10 = v4;
        if ( *(_DWORD *)(a1 + 8) )
        {
          memmove(*(void **)a1, v4, 4 * v7);
          v4 = *a2;
          v6 = *((unsigned int *)a2 + 2);
          v7 = v9;
          v10 = &(*a2)[v9];
        }
        v11 = 4 * v6;
        if ( v10 == &v4[v11] )
          goto LABEL_9;
      }
      memcpy((void *)(v7 + *(_QWORD *)a1), v10, v11 - v7);
LABEL_9:
      *(_DWORD *)(a1 + 8) = v8;
      *((_DWORD *)a2 + 2) = 0;
      return;
    }
    v12 = *(void **)a1;
    if ( v12 != (void *)(a1 + 16) )
    {
      _libc_free(v12, v5);
      v5 = *a2;
    }
    *(_QWORD *)a1 = v5;
    *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
    *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
    *a2 = v4;
    a2[1] = 0;
  }
}
