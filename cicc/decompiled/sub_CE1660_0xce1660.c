// Function: sub_CE1660
// Address: 0xce1660
//
__int64 __fastcall sub_CE1660(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49DD458;
  v2 = (_QWORD *)a1[23];
  if ( v2 != a1 + 25 )
    j_j___libc_free_0(v2, a1[25] + 1LL);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 216);
}
