// Function: sub_C43BD0
// Address: 0xc43bd0
//
_QWORD *__fastcall sub_C43BD0(_QWORD *a1, __int64 *a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdi
  unsigned __int64 v5; // rsi

  result = a1;
  v3 = *a1;
  v4 = *a2;
  v5 = ((unsigned __int64)*((unsigned int *)result + 2) + 63) >> 6;
  if ( v5 )
  {
    for ( result = 0; result != (_QWORD *)v5; result = (_QWORD *)((char *)result + 1) )
      *(_QWORD *)(v3 + 8LL * (_QWORD)result) |= *(_QWORD *)(v4 + 8LL * (_QWORD)result);
  }
  return result;
}
