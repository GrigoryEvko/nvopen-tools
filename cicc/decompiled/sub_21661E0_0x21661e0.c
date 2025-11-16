// Function: sub_21661E0
// Address: 0x21661e0
//
_QWORD *__fastcall sub_21661E0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  _QWORD *v4; // rax

  v3 = sub_1632FA0(*(_QWORD *)(a3 + 40));
  v4 = (_QWORD *)sub_22077B0(32);
  if ( v4 )
  {
    v4[1] = v3;
    v4[2] = a2 + 960;
    v4[3] = a2 + 1656;
    *v4 = &unk_4A02DA8;
  }
  *a1 = v4;
  return a1;
}
