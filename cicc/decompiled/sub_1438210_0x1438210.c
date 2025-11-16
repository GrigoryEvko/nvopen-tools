// Function: sub_1438210
// Address: 0x1438210
//
__int64 __fastcall sub_1438210(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49EB780;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 16);
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 168);
}
