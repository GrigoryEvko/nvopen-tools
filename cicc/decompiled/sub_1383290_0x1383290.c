// Function: sub_1383290
// Address: 0x1383290
//
__int64 __fastcall sub_1383290(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49E8DF0;
  if ( v1 )
  {
    sub_1383070(v1);
    j_j___libc_free_0(v1, 56);
  }
  return sub_16367B0(a1);
}
