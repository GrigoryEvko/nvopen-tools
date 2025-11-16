// Function: sub_2F62660
// Address: 0x2f62660
//
__int64 *__fastcall sub_2F62660(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax

  if ( !*(_BYTE *)(a1 + 692) )
    return sub_C8CC70(a1 + 664, a2, (__int64)a3, a4, a5, a6);
  result = *(__int64 **)(a1 + 672);
  a4 = *(unsigned int *)(a1 + 684);
  a3 = &result[a4];
  if ( result == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 680) )
      return sub_C8CC70(a1 + 664, a2, (__int64)a3, a4, a5, a6);
    *(_DWORD *)(a1 + 684) = a4 + 1;
    *a3 = a2;
    ++*(_QWORD *)(a1 + 664);
  }
  else
  {
    while ( a2 != *result )
    {
      if ( a3 == ++result )
        goto LABEL_7;
    }
  }
  return result;
}
