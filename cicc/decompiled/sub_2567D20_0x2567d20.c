// Function: sub_2567D20
// Address: 0x2567d20
//
_QWORD *__fastcall sub_2567D20(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx
  _QWORD *i; // rdx

  result = *(_QWORD **)(a1 + 8);
  v2 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[v2]; i != result; ++result )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
