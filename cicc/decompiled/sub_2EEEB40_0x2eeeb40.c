// Function: sub_2EEEB40
// Address: 0x2eeeb40
//
__int64 __fastcall sub_2EEEB40(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A2A3B8;
  v2 = a1[25];
  if ( (_QWORD *)v2 != a1 + 27 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
