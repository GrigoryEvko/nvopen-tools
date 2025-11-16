// Function: sub_5D28E0
// Address: 0x5d28e0
//
__int64 sub_5D28E0()
{
  _QWORD *v0; // r12
  __int64 *v1; // rbx
  __int64 i; // rdi
  __int64 **v3; // rax
  __int64 v4; // rdx

  v0 = &off_4A42AC0;
  v1 = (__int64 *)&unk_4CF7040;
  qword_4CF79B0 = sub_881A70(0xFFFFFFFFLL, 151, 3, 4);
  for ( i = qword_4CF79B0; ; i = qword_4CF79B0 )
  {
    v3 = (__int64 **)sub_881B20(i, *v0, 1);
    v4 = (__int64)*v3;
    v1[1] = (__int64)v0;
    v0 += 4;
    *v1 = v4;
    *v3 = v1;
    v1 += 2;
    if ( v1 == &qword_4CF79B0 )
      break;
  }
  if ( unk_4D04508 )
    sub_8539C0(&off_4A427E0);
  sub_8D0840(&qword_4CF6E18, 8, 0);
  sub_8D0840(&qword_4D04168, 8, 0);
  return sub_8D0840(&dword_4CF6E48, 4, 0);
}
