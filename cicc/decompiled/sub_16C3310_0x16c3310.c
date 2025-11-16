// Function: sub_16C3310
// Address: 0x16c3310
//
void *__fastcall sub_16C3310(__int64 a1, __int64 a2)
{
  size_t v3; // rdx
  size_t v4; // r13
  int v5; // r14d
  unsigned int v6; // ecx
  void *result; // rax
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
        v4 = 0;
        sub_16CD150(a1, a1 + 16, v3, 1);
        v3 = *(unsigned int *)(a2 + 8);
        v6 = *(_DWORD *)(a2 + 8);
      }
      else
      {
        v6 = *(_DWORD *)(a2 + 8);
        if ( *(_DWORD *)(a1 + 8) )
        {
          memmove(*(void **)a1, *(const void **)a2, v4);
          v3 = *(unsigned int *)(a2 + 8);
          v6 = *(_DWORD *)(a2 + 8);
        }
      }
      result = *(void **)a2;
      v8 = (const void *)(*(_QWORD *)a2 + v4);
      if ( v8 != (const void *)(*(_QWORD *)a2 + v3) )
        result = memcpy((void *)(v4 + *(_QWORD *)a1), v8, v6 - v4);
      goto LABEL_8;
    }
    if ( !*(_DWORD *)(a2 + 8) )
    {
LABEL_8:
      *(_DWORD *)(a1 + 8) = v5;
      return result;
    }
    result = memmove(*(void **)a1, *(const void **)a2, v3);
    *(_DWORD *)(a1 + 8) = v5;
  }
  return result;
}
