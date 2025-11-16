// Function: sub_2044F40
// Address: 0x2044f40
//
void *__fastcall sub_2044F40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // rdx
  size_t v8; // r13
  int v9; // r14d
  unsigned int v10; // ecx
  void *result; // rax
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
        v8 = 0;
        sub_16CD150(a1, (const void *)(a1 + 16), v7, 1, a5, a6);
        v7 = *(unsigned int *)(a2 + 8);
        v10 = *(_DWORD *)(a2 + 8);
      }
      else
      {
        v10 = *(_DWORD *)(a2 + 8);
        if ( *(_DWORD *)(a1 + 8) )
        {
          memmove(*(void **)a1, *(const void **)a2, v8);
          v7 = *(unsigned int *)(a2 + 8);
          v10 = *(_DWORD *)(a2 + 8);
        }
      }
      result = *(void **)a2;
      v12 = (const void *)(*(_QWORD *)a2 + v8);
      if ( v12 != (const void *)(*(_QWORD *)a2 + v7) )
        result = memcpy((void *)(v8 + *(_QWORD *)a1), v12, v10 - v8);
      goto LABEL_8;
    }
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_8:
      *(_DWORD *)(a1 + 8) = v9;
      return result;
    }
    result = memmove(*(void **)a1, *(const void **)a2, v7);
    *(_DWORD *)(a1 + 8) = v9;
  }
  return result;
}
