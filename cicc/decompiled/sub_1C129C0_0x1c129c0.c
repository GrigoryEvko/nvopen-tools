// Function: sub_1C129C0
// Address: 0x1c129c0
//
__int64 __fastcall sub_1C129C0(_QWORD *a1)
{
  __int64 v1; // r13
  __int64 v3; // rdi

  v1 = a1[20];
  *a1 = &unk_49F7560;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 56);
    *(_QWORD *)v1 = &unk_49F7548;
    if ( v3 )
      j_j___libc_free_0(v3, *(_QWORD *)(v1 + 72) - v3);
    sub_1C12880(*(_QWORD **)(v1 + 24));
    j_j___libc_free_0(v1, 88);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 168);
}
