// Function: sub_ECB2C0
// Address: 0xecb2c0
//
__int64 sub_ECB2C0()
{
  __int64 v0; // rax
  __int64 v1; // r12

  v0 = sub_22077B0(24);
  v1 = v0;
  if ( v0 )
  {
    sub_ECE400(v0);
    *(_BYTE *)(v1 + 16) = 1;
    *(_QWORD *)v1 = off_497ACE0;
  }
  return v1;
}
