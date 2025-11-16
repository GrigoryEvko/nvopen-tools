// Function: sub_D57810
// Address: 0xd57810
//
__int64 __fastcall sub_D57810(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49DE118;
  v2 = (_QWORD *)a1[23];
  if ( v2 != a1 + 25 )
    j_j___libc_free_0(v2, a1[25] + 1LL);
  *a1 = &unk_49DE2C8;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 216);
}
