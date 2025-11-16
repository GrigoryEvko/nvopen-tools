// Function: sub_228ADD0
// Address: 0x228add0
//
void __fastcall sub_228ADD0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A08D28;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
