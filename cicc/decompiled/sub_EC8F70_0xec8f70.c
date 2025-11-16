// Function: sub_EC8F70
// Address: 0xec8f70
//
_QWORD *sub_EC8F70()
{
  __int64 v0; // rax
  _QWORD *v1; // r12

  v0 = sub_22077B0(32);
  v1 = (_QWORD *)v0;
  if ( v0 )
  {
    sub_ECE400(v0);
    v1[3] = 0;
    *v1 = off_497ACB8;
  }
  return v1;
}
