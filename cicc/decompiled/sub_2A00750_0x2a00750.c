// Function: sub_2A00750
// Address: 0x2a00750
//
__int64 *__fastcall sub_2A00750(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *result; // rax
  __int64 *v4; // r12
  __int64 *v5; // r13
  __int64 *i; // rbx
  __int64 v7; // rsi

  result = *(__int64 **)(a1 + 56);
  v4 = (__int64 *)*result;
  if ( *result )
  {
    v5 = &a2[a3];
    for ( i = a2; v5 != i; result = sub_D4F330(v4, v7, *(_QWORD *)(a1 + 32)) )
      v7 = *i++;
  }
  return result;
}
