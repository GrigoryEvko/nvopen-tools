// Function: sub_ECFCE0
// Address: 0xecfce0
//
_QWORD *sub_ECFCE0()
{
  __int64 v0; // rax
  _QWORD *v1; // r12

  v0 = sub_22077B0(40);
  v1 = (_QWORD *)v0;
  if ( v0 )
  {
    sub_ECE400(v0);
    v1[3] = 0;
    *v1 = off_497AD58;
    v1[4] = 0;
  }
  return v1;
}
