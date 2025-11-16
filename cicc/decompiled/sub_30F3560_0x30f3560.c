// Function: sub_30F3560
// Address: 0x30f3560
//
_QWORD *__fastcall sub_30F3560(char *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rax
  _QWORD *result; // rax

  v4 = sub_DCF3A0(a3, a1, 0);
  if ( sub_D96A50(v4) || *(_WORD *)(v4 + 24) || (result = sub_DE5CD0(a3, v4)) == 0 )
  {
    v5 = (unsigned int)qword_5031488;
    v6 = sub_D95540(a2);
    return sub_DA2C50((__int64)a3, v6, v5, 0);
  }
  return result;
}
