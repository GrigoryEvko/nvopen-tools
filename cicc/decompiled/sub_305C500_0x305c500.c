// Function: sub_305C500
// Address: 0x305c500
//
_QWORD *__fastcall sub_305C500(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12

  v2 = sub_22077B0(0x118u);
  v3 = (_QWORD *)v2;
  if ( v2 )
  {
    sub_2FF0750(v2, a1, a2);
    *v3 = off_4A30C58;
  }
  return v3;
}
