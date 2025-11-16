// Function: sub_ECFAB0
// Address: 0xecfab0
//
__int64 sub_ECFAB0()
{
  __int64 v0; // rax
  __int64 v1; // r12

  v0 = sub_22077B0(40);
  v1 = v0;
  if ( v0 )
  {
    sub_ECE400(v0);
    *(_QWORD *)(v1 + 24) = 0;
    *(_QWORD *)v1 = off_497AD30;
    *(_QWORD *)(v1 + 32) = 0;
    *(_BYTE *)(v1 + 16) = 1;
  }
  return v1;
}
