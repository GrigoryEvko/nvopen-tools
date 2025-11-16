// Function: sub_15CEB50
// Address: 0x15ceb50
//
_QWORD *__fastcall sub_15CEB50(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[9 * v1]; i != result; result += 9 )
  {
    if ( result )
      *result = -8;
  }
  return result;
}
