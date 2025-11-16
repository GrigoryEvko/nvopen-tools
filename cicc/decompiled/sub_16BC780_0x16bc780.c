// Function: sub_16BC780
// Address: 0x16bc780
//
__int64 __fastcall sub_16BC780(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49EF2C0;
  v2 = (_QWORD *)a1[1];
  if ( v2 != a1 + 3 )
    j_j___libc_free_0(v2, a1[3] + 1LL);
  return j_j___libc_free_0(a1, 56);
}
