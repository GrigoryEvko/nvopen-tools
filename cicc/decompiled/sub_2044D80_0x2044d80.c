// Function: sub_2044D80
// Address: 0x2044d80
//
void __fastcall sub_2044D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  int v9; // r13d
  __int64 v10; // r14
  __int64 v11; // rdx
  const void *v12; // rsi

  if ( a1 != a2 )
  {
    v7 = *(unsigned int *)(a2 + 8);
    v8 = *(unsigned int *)(a1 + 8);
    v9 = *(_DWORD *)(a2 + 8);
    if ( v7 > v8 )
    {
      if ( v7 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        v10 = 0;
        sub_16CD150(a1, (const void *)(a1 + 16), v7, 20, a5, a6);
        v7 = *(unsigned int *)(a2 + 8);
      }
      else
      {
        v10 = 20 * v8;
        if ( *(_DWORD *)(a1 + 8) )
        {
          memmove(*(void **)a1, *(const void **)a2, 20 * v8);
          v7 = *(unsigned int *)(a2 + 8);
        }
      }
      v11 = 20 * v7;
      v12 = (const void *)(*(_QWORD *)a2 + v10);
      if ( v12 != (const void *)(v11 + *(_QWORD *)a2) )
        memcpy((void *)(v10 + *(_QWORD *)a1), v12, v11 - v10);
      goto LABEL_8;
    }
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_8:
      *(_DWORD *)(a1 + 8) = v9;
      return;
    }
    memmove(*(void **)a1, *(const void **)a2, 20 * v7);
    *(_DWORD *)(a1 + 8) = v9;
  }
}
