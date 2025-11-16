// Function: sub_DF4E70
// Address: 0xdf4e70
//
__int64 __fastcall sub_DF4E70(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49DEB50;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2, 1);
  sub_BB9280((__int64)a1);
  return j_j___libc_free_0(a1, 184);
}
