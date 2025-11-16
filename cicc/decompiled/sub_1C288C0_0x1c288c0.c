// Function: sub_1C288C0
// Address: 0x1c288c0
//
__int64 __fastcall sub_1C288C0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49F7658;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 200);
}
