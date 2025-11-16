// Function: sub_22E3410
// Address: 0x22e3410
//
void __fastcall sub_22E3410(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A0A288;
  v2 = a1[22];
  if ( (_QWORD *)v2 != a1 + 24 )
    j_j___libc_free_0(v2);
  *a1 = &unk_4A0A340;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
