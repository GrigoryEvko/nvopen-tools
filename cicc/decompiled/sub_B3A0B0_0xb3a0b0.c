// Function: sub_B3A0B0
// Address: 0xb3a0b0
//
__int64 __fastcall sub_B3A0B0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49DA308;
  v2 = (_QWORD *)a1[23];
  if ( v2 != a1 + 25 )
    j_j___libc_free_0(v2, a1[25] + 1LL);
  *a1 = &unk_49DAF80;
  sub_BB9100(a1);
  return j_j___libc_free_0(a1, 216);
}
