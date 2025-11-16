// Function: sub_2304350
// Address: 0x2304350
//
_QWORD *__fastcall sub_2304350(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD v5[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_3108400(v5, a2 + 8);
  v2 = v5[0];
  v3 = (_QWORD *)sub_22077B0(0x10u);
  if ( v3 )
  {
    v3[1] = v2;
    *v3 = &unk_4A159D0;
  }
  *a1 = v3;
  return a1;
}
