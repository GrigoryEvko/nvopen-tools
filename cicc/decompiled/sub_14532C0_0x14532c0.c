// Function: sub_14532C0
// Address: 0x14532c0
//
void __fastcall sub_14532C0(__int64 a1, char **a2)
{
  char *v4; // r13
  char *v5; // rsi
  void *v6; // rdi
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  int v9; // r14d
  __int64 v10; // r15
  char *v11; // rsi
  __int64 v12; // rdx

  if ( (char **)a1 != a2 )
  {
    v4 = (char *)(a2 + 2);
    v5 = *a2;
    if ( v5 == v4 )
    {
      v7 = *((unsigned int *)a2 + 2);
      v8 = *(unsigned int *)(a1 + 8);
      v9 = *((_DWORD *)a2 + 2);
      if ( v7 <= v8 )
      {
        if ( *((_DWORD *)a2 + 2) )
          memmove(*(void **)a1, v5, 8 * v7);
      }
      else
      {
        if ( v7 > *(unsigned int *)(a1 + 12) )
        {
          *(_DWORD *)(a1 + 8) = 0;
          sub_16CD150(a1, a1 + 16, v7, 8);
          v4 = *a2;
          v7 = *((unsigned int *)a2 + 2);
          v8 = 0;
          v11 = *a2;
        }
        else
        {
          v10 = 8 * v8;
          v11 = v4;
          if ( *(_DWORD *)(a1 + 8) )
          {
            memmove(*(void **)a1, v4, 8 * v8);
            v4 = *a2;
            v7 = *((unsigned int *)a2 + 2);
            v8 = v10;
            v11 = &(*a2)[v10];
          }
        }
        v12 = 8 * v7;
        if ( v11 != &v4[v12] )
          memcpy((void *)(v8 + *(_QWORD *)a1), v11, v12 - v8);
      }
      *(_DWORD *)(a1 + 8) = v9;
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v6 = *(void **)a1;
      if ( v6 != (void *)(a1 + 16) )
      {
        _libc_free((unsigned __int64)v6);
        v5 = *a2;
      }
      *(_QWORD *)a1 = v5;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
      *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
      *a2 = v4;
      a2[1] = 0;
    }
  }
}
