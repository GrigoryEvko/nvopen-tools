// Function: sub_155CDB0
// Address: 0x155cdb0
//
__int64 __fastcall sub_155CDB0(_QWORD *a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rdi

  *a1 = &unk_49ECE78;
  v2 = (_QWORD *)a1[7];
  if ( v2 != a1 + 9 )
    j_j___libc_free_0(v2, a1[9] + 1LL);
  v3 = (_QWORD *)a1[3];
  if ( v3 != a1 + 5 )
    j_j___libc_free_0(v3, a1[5] + 1LL);
  nullsub_545();
  return j_j___libc_free_0(a1, 88);
}
