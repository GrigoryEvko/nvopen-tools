// Function: sub_3255870
// Address: 0x3255870
//
_QWORD *__fastcall sub_3255870(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rbx

  v1 = (_QWORD *)sub_22077B0(0x10u);
  v2 = v1;
  if ( v1 )
  {
    *v1 = 0;
    v1[1] = 0;
    sub_34E6210(v1);
    *v2 = off_4A35EC8;
  }
  *a1 = v2;
  return a1;
}
