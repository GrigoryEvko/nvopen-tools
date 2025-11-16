// Function: sub_2D2B4E0
// Address: 0x2d2b4e0
//
_QWORD *__fastcall sub_2D2B4E0(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[2 * v1]; result != i; result += 2 )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
