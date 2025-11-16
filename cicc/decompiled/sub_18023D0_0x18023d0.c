// Function: sub_18023D0
// Address: 0x18023d0
//
__int64 __fastcall sub_18023D0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49F08F0;
  j___libc_free_0(a1[97]);
  j___libc_free_0(a1[93]);
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 800);
}
