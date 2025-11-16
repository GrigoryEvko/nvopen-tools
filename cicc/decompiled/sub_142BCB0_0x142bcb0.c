// Function: sub_142BCB0
// Address: 0x142bcb0
//
__int64 __fastcall sub_142BCB0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49EB478;
  v2 = a1[5];
  if ( v2 )
    j_j___libc_free_0(v2, a1[7] - v2);
  return j_j___libc_free_0(a1, 64);
}
