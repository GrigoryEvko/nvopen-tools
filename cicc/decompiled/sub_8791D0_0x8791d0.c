// Function: sub_8791D0
// Address: 0x8791d0
//
_QWORD *__fastcall sub_8791D0(_QWORD *a1)
{
  __int64 v1; // rcx
  _QWORD *result; // rax
  _QWORD *v3; // rdx

  v1 = a1[1];
  result = *(_QWORD **)(*a1 + 32LL);
  if ( result == a1 )
  {
    *(_QWORD *)(*a1 + 32LL) = v1;
    a1[1] = 0;
  }
  else
  {
    do
    {
      v3 = result;
      result = (_QWORD *)result[1];
    }
    while ( result != a1 );
    v3[1] = v1;
    a1[1] = 0;
  }
  return result;
}
