// Function: sub_D840D0
// Address: 0xd840d0
//
__int64 __fastcall sub_D840D0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rdi

  v1 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = 0;
  if ( v1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 64), 16LL * *(unsigned int *)(v1 + 80), 8);
    v2 = *(_QWORD *)(v1 + 8);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 8);
      if ( v3 )
        j_j___libc_free_0(v3, *(_QWORD *)(v2 + 24) - v3);
      j_j___libc_free_0(v2, 88);
    }
    j_j___libc_free_0(v1, 88);
  }
  return 0;
}
