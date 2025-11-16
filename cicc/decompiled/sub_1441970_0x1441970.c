// Function: sub_1441970
// Address: 0x1441970
//
__int64 __fastcall sub_1441970(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rdi

  v1 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = 0;
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 8);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 8);
      if ( v3 )
        j_j___libc_free_0(v3, *(_QWORD *)(v2 + 24) - v3);
      j_j___libc_free_0(v2, 72);
    }
    j_j___libc_free_0(v1, 56);
  }
  return 0;
}
