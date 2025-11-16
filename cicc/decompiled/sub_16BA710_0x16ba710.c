// Function: sub_16BA710
// Address: 0x16ba710
//
_QWORD *sub_16BA710()
{
  _QWORD *v0; // r8

  v0 = (_QWORD *)sub_22077B0(112);
  if ( v0 )
  {
    memset(v0, 0, 0x70u);
    v0[7] = v0 + 5;
    v0[8] = v0 + 5;
  }
  return v0;
}
