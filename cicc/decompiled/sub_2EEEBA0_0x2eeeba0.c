// Function: sub_2EEEBA0
// Address: 0x2eeeba0
//
void __fastcall sub_2EEEBA0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A2A3B8;
  v2 = a1[25];
  if ( (_QWORD *)v2 != a1 + 27 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
