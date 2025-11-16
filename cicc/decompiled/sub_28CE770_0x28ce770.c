// Function: sub_28CE770
// Address: 0x28ce770
//
_QWORD *__fastcall sub_28CE770(__int64 a1)
{
  __int64 v1; // rcx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[7 * v1]; i != result; result += 7 )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
