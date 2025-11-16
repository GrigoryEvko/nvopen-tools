// Function: sub_2A65660
// Address: 0x2a65660
//
unsigned __int64 __fastcall sub_2A65660(__int64 a1, size_t a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  size_t v9; // rdx
  void *v10; // rdi

  if ( *(_QWORD *)(a1 + 16) >= a2 )
  {
    result = *(_QWORD *)(a1 + 8);
    v9 = a2;
    if ( result <= a2 )
      v9 = *(_QWORD *)(a1 + 8);
    if ( v9 )
    {
      memset(*(void **)a1, a3, v9);
      result = *(_QWORD *)(a1 + 8);
    }
    if ( result < a2 )
    {
      v10 = (void *)(*(_QWORD *)a1 + result);
      if ( v10 != (void *)(a2 + *(_QWORD *)a1) )
      {
        result = (unsigned __int64)memset(v10, a3, a2 - result);
        *(_QWORD *)(a1 + 8) = a2;
        return result;
      }
    }
LABEL_7:
    *(_QWORD *)(a1 + 8) = a2;
    return result;
  }
  *(_QWORD *)(a1 + 8) = 0;
  result = sub_C8D290(a1, (const void *)(a1 + 24), a2, 1u, a5, a6);
  if ( !a2 )
    goto LABEL_7;
  result = (unsigned __int64)memset(*(void **)a1, a3, a2);
  *(_QWORD *)(a1 + 8) = a2;
  return result;
}
