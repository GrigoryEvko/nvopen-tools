// Function: sub_DE5CD0
// Address: 0xde5cd0
//
_QWORD *__fastcall sub_DE5CD0(__int64 *a1, __int64 a2)
{
  _QWORD **v3; // rbx
  int v4; // eax
  __int64 v5; // rax

  if ( sub_D96A50(a2) )
    return (_QWORD *)sub_D970F0((__int64)a1);
  v3 = (_QWORD **)sub_D95540(a2);
  v4 = sub_BCB060((__int64)v3);
  v5 = sub_BCD140(*v3, v4 + 1);
  return sub_DE5A20(a1, a2, v5, 0);
}
