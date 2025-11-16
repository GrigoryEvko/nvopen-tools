// Function: sub_C09870
// Address: 0xc09870
//
__int64 __fastcall sub_C09870(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13

  v2 = a1[22];
  *a1 = off_49DB3B8;
  if ( v2 )
  {
    sub_C08A70(v2, a2);
    j_j___libc_free_0(v2, 2320);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
