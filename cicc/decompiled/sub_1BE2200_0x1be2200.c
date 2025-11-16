// Function: sub_1BE2200
// Address: 0x1be2200
//
void __fastcall sub_1BE2200(__int64 a1, char **a2, __int64 a3, __int64 a4, int a5, int a6)
{
  char *v8; // r13
  char *v9; // rsi
  void *v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  int v13; // r14d
  __int64 v14; // r15
  char *v15; // rsi
  __int64 v16; // rdx

  if ( (char **)a1 != a2 )
  {
    v8 = (char *)(a2 + 2);
    v9 = *a2;
    if ( v9 == v8 )
    {
      v11 = *((unsigned int *)a2 + 2);
      v12 = *(unsigned int *)(a1 + 8);
      v13 = *((_DWORD *)a2 + 2);
      if ( v11 <= v12 )
      {
        if ( *((_DWORD *)a2 + 2) )
          memmove(*(void **)a1, v9, 8 * v11);
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          *(_DWORD *)(a1 + 8) = 0;
          sub_16CD150(a1, (const void *)(a1 + 16), v11, 8, a5, a6);
          v8 = *a2;
          v11 = *((unsigned int *)a2 + 2);
          v12 = 0;
          v15 = *a2;
        }
        else
        {
          v14 = 8 * v12;
          v15 = v8;
          if ( *(_DWORD *)(a1 + 8) )
          {
            memmove(*(void **)a1, v8, 8 * v12);
            v8 = *a2;
            v11 = *((unsigned int *)a2 + 2);
            v12 = v14;
            v15 = &(*a2)[v14];
          }
        }
        v16 = 8 * v11;
        if ( v15 != &v8[v16] )
          memcpy((void *)(v12 + *(_QWORD *)a1), v15, v16 - v12);
      }
      *(_DWORD *)(a1 + 8) = v13;
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v10 = *(void **)a1;
      if ( v10 != (void *)(a1 + 16) )
      {
        _libc_free((unsigned __int64)v10);
        v9 = *a2;
      }
      *(_QWORD *)a1 = v9;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
      *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
      *a2 = v8;
      a2[1] = 0;
    }
  }
}
