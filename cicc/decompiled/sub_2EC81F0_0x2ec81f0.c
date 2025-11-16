// Function: sub_2EC81F0
// Address: 0x2ec81f0
//
_QWORD *__fastcall sub_2EC81F0(_QWORD *a1)
{
  _QWORD *v1; // rax

  v1 = (_QWORD *)sub_22077B0(0x18u);
  if ( v1 )
  {
    v1[1] = 0;
    *v1 = off_4A29F08;
    v1[2] = 0;
  }
  *a1 = v1;
  return a1;
}
