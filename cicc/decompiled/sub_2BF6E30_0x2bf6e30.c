// Function: sub_2BF6E30
// Address: 0x2bf6e30
//
_QWORD *__fastcall sub_2BF6E30(__int64 a1, _QWORD *a2)
{
  _QWORD *result; // rax
  __int64 v3; // rcx

  result = *(_QWORD **)a1;
  v3 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        *a2 = *result;
        a2[1] = result[1];
        a2[2] = result[2];
        a2[3] = result[3];
        a2[4] = result[4];
      }
      result += 5;
      a2 += 5;
    }
    while ( (_QWORD *)v3 != result );
  }
  return result;
}
