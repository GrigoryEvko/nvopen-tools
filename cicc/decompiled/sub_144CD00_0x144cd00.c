// Function: sub_144CD00
// Address: 0x144cd00
//
__int64 __fastcall sub_144CD00(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49EBF28;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 192);
}
