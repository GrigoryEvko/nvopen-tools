// Function: sub_1060950
// Address: 0x1060950
//
_QWORD *__fastcall sub_1060950(_QWORD *a1)
{
  _OWORD *v1; // rax
  _QWORD *v2; // rbx

  v1 = (_OWORD *)sub_22077B0(48);
  v2 = v1;
  if ( v1 )
  {
    *v1 = 0;
    v1[1] = 0;
    v1[2] = 0;
    sub_E3FC50((__int64)v1);
    *v2 = &off_49E5E50;
  }
  *a1 = v2;
  return a1;
}
