// Function: sub_13997A0
// Address: 0x13997a0
//
__int64 __fastcall sub_13997A0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49E90F0;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  return sub_1636790(a1);
}
