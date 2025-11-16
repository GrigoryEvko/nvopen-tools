// Function: sub_D1D810
// Address: 0xd1d810
//
__int64 __fastcall sub_D1D810(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[22];
  *a1 = &unk_49DDE78;
  if ( v1 )
  {
    sub_D1D5E0(v1);
    j_j___libc_free_0(v1, 360);
  }
  return sub_BB9260((__int64)a1);
}
