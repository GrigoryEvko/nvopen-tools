// Function: sub_15E90F0
// Address: 0x15e90f0
//
__int64 __fastcall sub_15E90F0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49ED328;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE120;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 200);
}
