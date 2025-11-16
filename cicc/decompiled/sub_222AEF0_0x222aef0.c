// Function: sub_222AEF0
// Address: 0x222aef0
//
_QWORD *__fastcall sub_222AEF0(_QWORD *a1, __int64 a2, __int64 a3)
{
  if ( sub_2207CD0(a1 + 13) )
    return a1;
  if ( !(a3 | a2) )
  {
    a1[20] = 1;
    return a1;
  }
  if ( !a2 || a3 <= 0 )
    return a1;
  a1[19] = a2;
  a1[20] = a3;
  return a1;
}
