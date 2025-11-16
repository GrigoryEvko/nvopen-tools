// Function: sub_229BB50
// Address: 0x229bb50
//
void __fastcall sub_229BB50(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A08ED0;
  v2 = a1[22];
  if ( (_QWORD *)v2 != a1 + 24 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
