// Function: sub_B7C7E0
// Address: 0xb7c7e0
//
__int64 __fastcall sub_B7C7E0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49DA3F8;
  v2 = (_QWORD *)a1[1];
  if ( v2 != a1 + 3 )
    j_j___libc_free_0(v2, a1[3] + 1LL);
  return j_j___libc_free_0(a1, 56);
}
