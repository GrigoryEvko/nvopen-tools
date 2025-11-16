// Function: sub_144C900
// Address: 0x144c900
//
__int64 __fastcall sub_144C900(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = off_49EC088;
  v2 = (_QWORD *)a1[20];
  if ( v2 != a1 + 22 )
    j_j___libc_free_0(v2, a1[22] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
