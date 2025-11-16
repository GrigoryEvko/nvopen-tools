// Function: sub_B14410
// Address: 0xb14410
//
__int64 *__fastcall sub_B14410(__int64 a1, __int64 a2, char a3)
{
  __int64 *v3; // rcx
  __int64 *result; // rax
  __int64 *v5; // rdx
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi

  v3 = (__int64 *)(a1 + 8);
  if ( a3 )
    v3 = *(__int64 **)(a1 + 16);
  result = *(__int64 **)(a2 + 16);
  v5 = (__int64 *)(a2 + 8);
  if ( (__int64 *)(a2 + 8) != result )
  {
    do
    {
      result[2] = a1;
      result = (__int64 *)result[1];
    }
    while ( v5 != result );
    result = *(__int64 **)(a2 + 16);
  }
  if ( v5 != v3 && v5 != result )
  {
    v6 = *(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*result & 0xFFFFFFFFFFFFFFF8LL) + 8) = v5;
    *(_QWORD *)(a2 + 8) = *(_QWORD *)(a2 + 8) & 7LL | *result & 0xFFFFFFFFFFFFFFF8LL;
    v7 = *v3;
    *(_QWORD *)(v6 + 8) = v3;
    v7 &= 0xFFFFFFFFFFFFFFF8LL;
    *result = v7 | *result & 7;
    *(_QWORD *)(v7 + 8) = result;
    result = (__int64 *)(v6 | *v3 & 7);
    *v3 = (__int64)result;
  }
  return result;
}
