// Function: sub_217EC60
// Address: 0x217ec60
//
_QWORD *__fastcall sub_217EC60(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[5 * v1]; i != result; result += 5 )
  {
    if ( result )
      *result = -8;
  }
  return result;
}
