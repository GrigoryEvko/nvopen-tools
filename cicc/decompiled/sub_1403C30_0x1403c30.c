// Function: sub_1403C30
// Address: 0x1403c30
//
__int64 __fastcall sub_1403C30(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49EAD28;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EAEF0;
  return sub_16366C0(a1);
}
