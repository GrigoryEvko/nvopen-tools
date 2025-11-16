// Function: sub_9B6660
// Address: 0x9b6660
//
void __fastcall sub_9B6660(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  int v5; // r13d
  __int64 v6; // r14
  __int64 v7; // rdx
  const void *v8; // rsi

  if ( a1 != a2 )
  {
    v3 = *(unsigned int *)(a2 + 8);
    v4 = *(unsigned int *)(a1 + 8);
    v5 = *(_DWORD *)(a2 + 8);
    if ( v3 > v4 )
    {
      if ( v3 > *(unsigned int *)(a1 + 12) )
      {
        *(_DWORD *)(a1 + 8) = 0;
        v6 = 0;
        sub_C8D5F0(a1, a1 + 16, v3, 4);
        v3 = *(unsigned int *)(a2 + 8);
      }
      else
      {
        v6 = 4 * v4;
        if ( *(_DWORD *)(a1 + 8) )
        {
          memmove(*(void **)a1, *(const void **)a2, 4 * v4);
          v3 = *(unsigned int *)(a2 + 8);
        }
      }
      v7 = 4 * v3;
      v8 = (const void *)(*(_QWORD *)a2 + v6);
      if ( v8 != (const void *)(v7 + *(_QWORD *)a2) )
        memcpy((void *)(v6 + *(_QWORD *)a1), v8, v7 - v6);
      goto LABEL_8;
    }
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_8:
      *(_DWORD *)(a1 + 8) = v5;
      return;
    }
    memmove(*(void **)a1, *(const void **)a2, 4 * v3);
    *(_DWORD *)(a1 + 8) = v5;
  }
}
