// Function: sub_D0FB90
// Address: 0xd0fb90
//
__int64 __fastcall sub_D0FB90(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[22];
  *a1 = &unk_49DDCE8;
  if ( v1 )
  {
    sub_D0FA70(v1);
    j_j___libc_free_0(v1, 72);
  }
  return sub_BB9260((__int64)a1);
}
