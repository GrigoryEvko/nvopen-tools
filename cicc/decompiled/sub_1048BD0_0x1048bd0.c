// Function: sub_1048BD0
// Address: 0x1048bd0
//
__int64 __fastcall sub_1048BD0(_QWORD *a1)
{
  __int64 v1; // r13
  __int64 v2; // r14

  v1 = a1[22];
  *a1 = &unk_49E5BC0;
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 16);
    if ( v2 )
    {
      sub_FDC110(*(__int64 **)(v1 + 16));
      j_j___libc_free_0(v2, 8);
    }
    j_j___libc_free_0(v1, 24);
  }
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 184);
}
