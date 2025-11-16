// Function: sub_30A4110
// Address: 0x30a4110
//
__int64 __fastcall sub_30A4110(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A31F08;
  v2 = a1[22];
  if ( (_QWORD *)v2 != a1 + 24 )
    j_j___libc_free_0(v2);
  *a1 = &unk_4A31FC0;
  return sub_BB9100((__int64)a1);
}
