// Function: sub_144C9C0
// Address: 0x144c9c0
//
__int64 __fastcall sub_144C9C0(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49EC348;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
