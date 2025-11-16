// Function: sub_1704480
// Address: 0x1704480
//
__int64 __fastcall sub_1704480(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_49EFFD8;
  j___libc_free_0(a1[279]);
  v2 = a1[20];
  if ( (_QWORD *)v2 != a1 + 22 )
    _libc_free(v2);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 2264);
}
