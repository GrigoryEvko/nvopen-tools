// Function: sub_2044890
// Address: 0x2044890
//
void __fastcall sub_2044890(__int64 a1, char **a2, __int64 a3, __int64 a4, int a5, int a6)
{
  char *v8; // r12
  char *v9; // rsi
  void *v10; // rdi
  size_t v11; // rdx
  size_t v12; // r14
  int v13; // r15d
  char *v14; // rsi
  unsigned int v15; // eax

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
          memmove(*(void **)a1, v9, v11);
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          *(_DWORD *)(a1 + 8) = 0;
          v12 = 0;
          sub_16CD150(a1, (const void *)(a1 + 16), v11, 1, a5, a6);
          v8 = *a2;
          v11 = *((unsigned int *)a2 + 2);
          v14 = *a2;
          v15 = *((_DWORD *)a2 + 2);
        }
        else
        {
          v14 = v8;
          v15 = *((_DWORD *)a2 + 2);
          if ( *(_DWORD *)(a1 + 8) )
          {
            memmove(*(void **)a1, v8, v12);
            v8 = *a2;
            v11 = *((unsigned int *)a2 + 2);
            v14 = &(*a2)[v12];
            v15 = *((_DWORD *)a2 + 2);
          }
        }
        if ( v14 != &v8[v11] )
          memcpy((void *)(v12 + *(_QWORD *)a1), v14, v15 - v12);
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
