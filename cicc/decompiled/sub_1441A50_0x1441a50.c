// Function: sub_1441A50
// Address: 0x1441a50
//
__int64 __fastcall sub_1441A50(_QWORD *a1)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rdi

  v2 = a1[20];
  *a1 = &unk_49EBA28;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 8);
    if ( v3 )
    {
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 )
        j_j___libc_free_0(v4, *(_QWORD *)(v3 + 24) - v4);
      j_j___libc_free_0(v3, 72);
    }
    j_j___libc_free_0(v2, 56);
  }
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 168);
}
