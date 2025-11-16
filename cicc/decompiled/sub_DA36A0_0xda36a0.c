// Function: sub_DA36A0
// Address: 0xda36a0
//
_QWORD *__fastcall sub_DA36A0(__int64 *a1, char a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rax

  v2 = (_QWORD *)sub_B2BE50(*a1);
  v3 = sub_BCB2A0(v2);
  if ( a2 )
    return sub_DA2C50((__int64)a1, v3, 1, 0);
  else
    return sub_DA2C50((__int64)a1, v3, 0, 0);
}
