// Function: sub_1D59CE0
// Address: 0x1d59ce0
//
__int64 __fastcall sub_1D59CE0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49F9CF0;
  v2 = (_QWORD *)a1[1];
  if ( v2 != a1 + 3 )
    j_j___libc_free_0(v2, a1[3] + 1LL);
  return j_j___libc_free_0(a1, 56);
}
