// Function: sub_732DB0
// Address: 0x732db0
//
_QWORD *__fastcall sub_732DB0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *result; // rax
  _QWORD *v5; // rdx

  result = *(_QWORD **)(a2 + 24);
  if ( result )
  {
    do
    {
      v5 = result;
      result = (_QWORD *)result[4];
    }
    while ( result );
    v5[4] = a1;
  }
  else
  {
    *(_QWORD *)(a2 + 24) = a1;
  }
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) = a2;
  if ( a3 )
  {
    for ( result = *(_QWORD **)(a2 + 48); result; result = (_QWORD *)result[7] )
    {
      if ( !result[5] )
        result[5] = a1;
    }
  }
  return result;
}
