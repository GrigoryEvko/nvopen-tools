// Function: sub_C7FCA0
// Address: 0xc7fca0
//
void *__fastcall sub_C7FCA0(__int64 a1, __int64 a2)
{
  size_t v3; // r13
  size_t v4; // r14
  __int64 v5; // rdx
  const void *v6; // rsi
  void *result; // rax
  __int64 v8; // rdx

  if ( a1 != a2 )
  {
    v3 = *(_QWORD *)(a2 + 8);
    v4 = *(_QWORD *)(a1 + 8);
    if ( v3 > v4 )
    {
      if ( v3 > *(_QWORD *)(a1 + 16) )
      {
        v8 = *(_QWORD *)(a2 + 8);
        v4 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        sub_C8D290(a1, a1 + 24, v8, 1);
        v5 = *(_QWORD *)(a2 + 8);
      }
      else
      {
        v5 = *(_QWORD *)(a2 + 8);
        if ( v4 )
        {
          memmove(*(void **)a1, *(const void **)a2, v4);
          v5 = *(_QWORD *)(a2 + 8);
        }
      }
      v6 = (const void *)(*(_QWORD *)a2 + v4);
      result = (void *)(v5 + *(_QWORD *)a2);
      if ( v6 != result )
        result = memcpy((void *)(v4 + *(_QWORD *)a1), v6, v5 - v4);
      goto LABEL_8;
    }
    if ( !v3 )
    {
LABEL_8:
      *(_QWORD *)(a1 + 8) = v3;
      return result;
    }
    result = memmove(*(void **)a1, *(const void **)a2, v3);
    *(_QWORD *)(a1 + 8) = v3;
  }
  return result;
}
