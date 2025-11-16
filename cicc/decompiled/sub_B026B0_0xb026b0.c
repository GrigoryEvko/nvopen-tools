// Function: sub_B026B0
// Address: 0xb026b0
//
_QWORD *__fastcall sub_B026B0(__int64 a1, __int64 a2)
{
  if ( !a1 || !a2 )
    return 0;
  if ( a1 == a2 )
    return (_QWORD *)a1;
  return sub_B01B40(a1, a2);
}
