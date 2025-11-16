// Function: sub_144CC20
// Address: 0x144cc20
//
__int64 __fastcall sub_144CC20(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49EC348;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 192);
}
