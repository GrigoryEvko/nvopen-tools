// Function: sub_C4F7F0
// Address: 0xc4f7f0
//
_QWORD *sub_C4F7F0()
{
  _QWORD *v0; // r8

  v0 = (_QWORD *)sub_22077B0(160);
  if ( v0 )
  {
    memset(v0, 0, 0xA0u);
    v0[10] = v0 + 12;
    v0[4] = v0 + 6;
    v0[5] = 0x400000000LL;
    v0[11] = 0x400000000LL;
    v0[18] = 0x1000000000LL;
  }
  return v0;
}
