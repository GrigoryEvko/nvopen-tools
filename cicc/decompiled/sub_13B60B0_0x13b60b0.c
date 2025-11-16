// Function: sub_13B60B0
// Address: 0x13b60b0
//
__int64 __fastcall sub_13B60B0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49EA120;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 192);
}
