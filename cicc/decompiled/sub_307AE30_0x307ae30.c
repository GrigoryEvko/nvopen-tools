// Function: sub_307AE30
// Address: 0x307ae30
//
__int64 __fastcall sub_307AE30(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A31850;
  v2 = a1[26];
  if ( (_QWORD *)v2 != a1 + 28 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
