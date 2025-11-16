// Function: sub_16890B0
// Address: 0x16890b0
//
_QWORD *sub_16890B0()
{
  __int64 v0; // rdi
  int v1; // edx
  int v2; // ecx
  int v3; // r8d
  int v4; // r9d
  _QWORD *v5; // r12
  char v7; // [rsp+0h] [rbp-10h]

  v0 = *((_QWORD *)sub_1689050() + 3);
  v5 = sub_1685080(v0, 40);
  if ( !v5 )
    sub_1683C30(v0, 40, v1, v2, v3, v4, v7);
  v5[4] = 0;
  *(_OWORD *)v5 = 0;
  *((_OWORD *)v5 + 1) = 0;
  if ( sub_1688DC0((pthread_mutex_t *)v5) )
    return v5;
  sub_16856A0(v5);
  return 0;
}
