// Function: sub_13F3CA0
// Address: 0x13f3ca0
//
__int64 __fastcall sub_13F3CA0(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi

  v2 = a1 + 30;
  *(v2 - 30) = off_49EAAB0;
  sub_16E7BC0(v2);
  v3 = (_QWORD *)a1[26];
  if ( v3 != a1 + 28 )
    j_j___libc_free_0(v3, a1[28] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
