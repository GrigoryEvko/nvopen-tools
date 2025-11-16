// Function: sub_13B5D70
// Address: 0x13b5d70
//
__int64 __fastcall sub_13B5D70(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49EA280;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
