// Function: sub_3568750
// Address: 0x3568750
//
void __fastcall sub_3568750(_QWORD *a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // r14
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // r13

  v1 = a1[10];
  while ( v1 )
  {
    v3 = v1;
    sub_35674C0(*(_QWORD **)(v1 + 24));
    v4 = *(_QWORD *)(v1 + 40);
    v1 = *(_QWORD *)(v1 + 16);
    if ( v4 )
      j_j___libc_free_0(v4);
    j_j___libc_free_0(v3);
  }
  v5 = (unsigned __int64 *)a1[6];
  v6 = (unsigned __int64 *)a1[5];
  if ( v5 != v6 )
  {
    do
    {
      v7 = *v6;
      if ( *v6 )
      {
        sub_3568740(*v6);
        j_j___libc_free_0(v7);
      }
      ++v6;
    }
    while ( v5 != v6 );
    v6 = (unsigned __int64 *)a1[5];
  }
  if ( v6 )
    j_j___libc_free_0((unsigned __int64)v6);
}
