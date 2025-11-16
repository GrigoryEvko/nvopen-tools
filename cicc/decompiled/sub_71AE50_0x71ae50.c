// Function: sub_71AE50
// Address: 0x71ae50
//
_QWORD *__fastcall sub_71AE50(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12

  while ( 1 )
  {
    v2 = (_QWORD *)sub_6EE5A0(a1);
    if ( !(unsigned int)sub_8D2E30(*v2) )
      break;
    v1 = sub_8D46C0(*v2);
    if ( !(unsigned int)sub_8D3410(v1) )
      break;
    a1 = sub_73DCD0(v2);
  }
  return v2;
}
