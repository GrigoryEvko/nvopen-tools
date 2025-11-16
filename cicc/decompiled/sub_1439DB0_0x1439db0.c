// Function: sub_1439DB0
// Address: 0x1439db0
//
__int64 __fastcall sub_1439DB0(_QWORD *a1)
{
  __int64 v1; // r13
  __int64 v2; // r14

  v1 = a1[20];
  *a1 = &unk_49EB830;
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 16);
    if ( v2 )
    {
      sub_1368A00(*(__int64 **)(v1 + 16));
      j_j___libc_free_0(v2, 8);
    }
    j_j___libc_free_0(v1, 24);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
