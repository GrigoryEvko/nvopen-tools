// Function: sub_23992F0
// Address: 0x23992f0
//
_QWORD *__fastcall sub_23992F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rax
  _QWORD *v4; // rbx

  v2 = *(_QWORD *)(a2 + 8);
  v3 = (_QWORD *)sub_22077B0(0x10u);
  v4 = v3;
  if ( v3 )
  {
    v3[1] = v2;
    *v3 = &unk_4A0AB10;
  }
  else if ( v2 )
  {
    sub_2398D90(v2 + 64);
    sub_2398F30(v2 + 32);
  }
  *a1 = v4;
  return a1;
}
