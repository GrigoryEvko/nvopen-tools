// Function: sub_2EC2120
// Address: 0x2ec2120
//
void __fastcall sub_2EC2120(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rdi

  v1 = a1 - 8;
  v2 = a1 + 32;
  qword_5021050[2] = 0;
  if ( *(_QWORD *)(v2 - 16) != v2 )
    _libc_free(*(_QWORD *)(v2 - 16));
  j_j___libc_free_0(v1);
}
