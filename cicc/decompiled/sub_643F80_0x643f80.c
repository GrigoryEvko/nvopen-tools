// Function: sub_643F80
// Address: 0x643f80
//
_QWORD *__fastcall sub_643F80(__int64 a1, _QWORD *a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rdx
  __int64 v4; // rdx

  result = *(_QWORD **)(a1 + 432);
  if ( result != a2 )
  {
    do
    {
      result[1] = 0;
      v3 = result;
      result = (_QWORD *)*result;
    }
    while ( result != a2 );
    *v3 = qword_4CFDE70;
    v4 = *(_QWORD *)(a1 + 432);
    *(_QWORD *)(a1 + 432) = result;
    qword_4CFDE70 = v4;
  }
  return result;
}
