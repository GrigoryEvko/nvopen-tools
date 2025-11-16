// Function: sub_2C961C0
// Address: 0x2c961c0
//
_QWORD *__fastcall sub_2C961C0(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  _QWORD *i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[4 * v1]; i != result; result += 4 )
  {
    if ( result )
      *result = -4096;
  }
  return result;
}
