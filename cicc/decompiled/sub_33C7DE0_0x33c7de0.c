// Function: sub_33C7DE0
// Address: 0x33c7de0
//
_QWORD *__fastcall sub_33C7DE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *result; // rax
  __int64 i; // rdx

  v2 = *(_QWORD *)(a1 + 24);
  result = *(_QWORD **)v2;
  for ( i = *(_QWORD *)v2 + 24LL * *(unsigned int *)(v2 + 8); (_QWORD *)i != result; result += 3 )
  {
    while ( *result != a2 )
    {
      result += 3;
      if ( (_QWORD *)i == result )
        return result;
    }
    *result = 0;
  }
  return result;
}
