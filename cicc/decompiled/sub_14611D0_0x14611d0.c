// Function: sub_14611D0
// Address: 0x14611d0
//
__int64 __fastcall sub_14611D0(_QWORD *a1)
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
  return sub_16366C0(a1);
}
