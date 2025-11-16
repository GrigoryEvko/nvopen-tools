// Function: sub_ED09D0
// Address: 0xed09d0
//
__int64 __fastcall sub_ED09D0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49E4BC8;
  v2 = (_QWORD *)a1[2];
  if ( v2 != a1 + 4 )
    j_j___libc_free_0(v2, a1[4] + 1LL);
  return j_j___libc_free_0(a1, 48);
}
