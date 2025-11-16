// Function: sub_1B68380
// Address: 0x1b68380
//
__int64 __fastcall sub_1B68380(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi

  *a1 = off_49853F8;
  v2 = (_QWORD *)a1[6];
  if ( v2 != a1 + 8 )
    j_j___libc_free_0(v2, a1[8] + 1LL);
  v3 = (_QWORD *)a1[2];
  if ( v3 != a1 + 4 )
    j_j___libc_free_0(v3, a1[4] + 1LL);
  return j_j___libc_free_0(a1, 80);
}
