// Function: sub_1461230
// Address: 0x1461230
//
__int64 __fastcall sub_1461230(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49EC5F0;
  if ( v1 )
  {
    sub_14602B0(v1);
    j_j___libc_free_0(v1, 1040);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 168);
}
