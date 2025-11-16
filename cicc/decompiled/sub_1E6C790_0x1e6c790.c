// Function: sub_1E6C790
// Address: 0x1e6c790
//
__int64 __fastcall sub_1E6C790(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_49FCB18;
  v2 = a1[5];
  if ( v2 )
    j_j___libc_free_0(v2, a1[7] - v2);
  return j_j___libc_free_0(a1, 64);
}
