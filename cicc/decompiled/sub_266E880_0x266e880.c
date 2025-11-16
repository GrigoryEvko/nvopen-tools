// Function: sub_266E880
// Address: 0x266e880
//
void *__fastcall sub_266E880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v7; // r13
  size_t v8; // r14
  __int64 v9; // rdx
  const void *v10; // rsi
  void *result; // rax
  __int64 v12; // rdx

  if ( a1 != a2 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v8 = *(_QWORD *)(a1 + 8);
    if ( v7 > v8 )
    {
      if ( v7 > *(_QWORD *)(a1 + 16) )
      {
        v12 = *(_QWORD *)(a2 + 8);
        v8 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        sub_C8D290(a1, (const void *)(a1 + 24), v12, 1u, a5, a6);
        v9 = *(_QWORD *)(a2 + 8);
      }
      else
      {
        v9 = *(_QWORD *)(a2 + 8);
        if ( v8 )
        {
          memmove(*(void **)a1, *(const void **)a2, v8);
          v9 = *(_QWORD *)(a2 + 8);
        }
      }
      v10 = (const void *)(*(_QWORD *)a2 + v8);
      result = (void *)(v9 + *(_QWORD *)a2);
      if ( v10 != result )
        result = memcpy((void *)(v8 + *(_QWORD *)a1), v10, v9 - v8);
      goto LABEL_8;
    }
    if ( !v7 )
    {
LABEL_8:
      *(_QWORD *)(a1 + 8) = v7;
      return result;
    }
    result = memmove(*(void **)a1, *(const void **)a2, v7);
    *(_QWORD *)(a1 + 8) = v7;
  }
  return result;
}
