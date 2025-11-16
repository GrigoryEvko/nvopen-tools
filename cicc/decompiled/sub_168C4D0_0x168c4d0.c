// Function: sub_168C4D0
// Address: 0x168c4d0
//
__int64 __fastcall sub_168C4D0(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi

  *a1 = &unk_49EE580;
  v2 = (_QWORD *)a1[8];
  if ( v2 != a1 + 10 )
    j_j___libc_free_0(v2, a1[10] + 1LL);
  v3 = (_QWORD *)a1[1];
  if ( v3 != a1 + 3 )
    j_j___libc_free_0(v3, a1[3] + 1LL);
  return j_j___libc_free_0(a1, 216);
}
