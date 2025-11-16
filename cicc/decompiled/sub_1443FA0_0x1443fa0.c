// Function: sub_1443FA0
// Address: 0x1443fa0
//
_QWORD *__fastcall sub_1443FA0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r13
  _QWORD *v4; // rbx

  v2 = sub_1443F20(a1[2], a2);
  if ( !v2 )
    return 0;
  v3 = (_QWORD *)v2;
  if ( a1 == (_QWORD *)v2 )
    return 0;
  while ( 1 )
  {
    v4 = (_QWORD *)v3[1];
    if ( !a1[4] )
      goto LABEL_10;
    if ( !(unsigned __int8)sub_1443560(a1, *v4 & 0xFFFFFFFFFFFFFFF8LL)
      || !(unsigned __int8)sub_1443560(a1, v4[4]) && v4[4] != a1[4] )
    {
      break;
    }
    v4 = (_QWORD *)v3[1];
LABEL_10:
    if ( a1 == v4 )
      break;
    v3 = v4;
  }
  if ( a2 == (*v3 & 0xFFFFFFFFFFFFFFF8LL) )
    return v3;
  return 0;
}
