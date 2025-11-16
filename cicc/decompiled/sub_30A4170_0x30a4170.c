// Function: sub_30A4170
// Address: 0x30a4170
//
void __fastcall sub_30A4170(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A31F08;
  v2 = a1[22];
  if ( (_QWORD *)v2 != a1 + 24 )
    j_j___libc_free_0(v2);
  *a1 = &unk_4A31FC0;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
