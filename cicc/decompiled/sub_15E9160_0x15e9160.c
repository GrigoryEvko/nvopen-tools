// Function: sub_15E9160
// Address: 0x15e9160
//
__int64 __fastcall sub_15E9160(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49ED280;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 200);
}
