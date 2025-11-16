// Function: sub_103CD50
// Address: 0x103cd50
//
__int64 __fastcall sub_103CD50(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[22];
  *a1 = &unk_49E5B18;
  if ( v1 )
  {
    sub_103C970(v1);
    j_j___libc_free_0(v1, 360);
  }
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 184);
}
