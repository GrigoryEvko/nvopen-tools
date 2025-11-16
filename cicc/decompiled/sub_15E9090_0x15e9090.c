// Function: sub_15E9090
// Address: 0x15e9090
//
__int64 __fastcall sub_15E9090(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49ED280;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
