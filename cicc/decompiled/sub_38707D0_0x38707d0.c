// Function: sub_38707D0
// Address: 0x38707d0
//
__int64 *__fastcall sub_38707D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 *result; // rax
  __int64 *i; // rcx
  __int64 v6; // rdx
  __int64 v7; // rsi

  v2 = a2 + 24;
  v3 = *(_QWORD *)(a2 + 32);
  if ( *(_QWORD *)(a1 + 280) == a2 + 24 )
  {
    v7 = v3 - 24;
    if ( !v3 )
      v7 = 0;
    sub_17050D0((__int64 *)(a1 + 264), v7);
  }
  result = *(__int64 **)(a1 + 336);
  for ( i = &result[*(unsigned int *)(a1 + 344)]; i != result; *(_QWORD *)(v6 + 16) = v3 )
  {
    while ( 1 )
    {
      v6 = *result;
      if ( v2 == *(_QWORD *)(*result + 16) )
        break;
      if ( i == ++result )
        return result;
    }
    ++result;
  }
  return result;
}
