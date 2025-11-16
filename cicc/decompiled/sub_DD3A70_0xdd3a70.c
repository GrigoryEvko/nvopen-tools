// Function: sub_DD3A70
// Address: 0xdd3a70
//
_QWORD *__fastcall sub_DD3A70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12

  v4 = sub_DD3750(a1, a2);
  if ( sub_D96A50(v4) )
    return (_QWORD *)v4;
  else
    return sub_DC5760(a1, v4, a3, 0);
}
