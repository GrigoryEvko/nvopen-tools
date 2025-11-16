// Function: sub_13832E0
// Address: 0x13832e0
//
__int64 __fastcall sub_13832E0(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49E8DF0;
  if ( v1 )
  {
    sub_1383070(v1);
    j_j___libc_free_0(v1, 56);
  }
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 168);
}
