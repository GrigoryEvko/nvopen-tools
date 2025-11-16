// Function: sub_7335F0
// Address: 0x7335f0
//
_QWORD *__fastcall sub_7335F0(_QWORD *a1, unsigned __int8 *a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12

  v2 = sub_726700(10);
  v2[7] = a1;
  v3 = v2;
  sub_730580((__int64)a1, (__int64)v2);
  *v3 = *a1;
  sub_732E60(a2, 0xDu, v3);
  return v3;
}
