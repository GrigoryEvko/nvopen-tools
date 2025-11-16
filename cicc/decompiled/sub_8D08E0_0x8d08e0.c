// Function: sub_8D08E0
// Address: 0x8d08e0
//
_QWORD *__fastcall sub_8D08E0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  result = (_QWORD *)qword_4F60540;
  if ( qword_4F60540 )
  {
    do
    {
      v2 = result[4];
      if ( v2 )
        *(_QWORD *)(a1 + v2) = result[1];
      result = (_QWORD *)*result;
    }
    while ( result );
  }
  return result;
}
