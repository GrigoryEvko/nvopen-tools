// Function: sub_3352FA0
// Address: 0x3352fa0
//
void __fastcall sub_3352FA0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = &off_4A362C0;
  v2 = a1[18];
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = a1[15];
  if ( v3 )
    j_j___libc_free_0(v3);
  v4 = a1[12];
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = a1[2];
  if ( v5 )
    j_j___libc_free_0(v5);
  j_j___libc_free_0((unsigned __int64)a1);
}
