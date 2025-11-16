// Function: sub_D84150
// Address: 0xd84150
//
__int64 __fastcall sub_D84150(_QWORD *a1)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rdi

  v2 = a1[22];
  *a1 = &unk_49DE690;
  if ( v2 )
  {
    sub_C7D6A0(*(_QWORD *)(v2 + 64), 16LL * *(unsigned int *)(v2 + 80), 8);
    v3 = *(_QWORD *)(v2 + 8);
    if ( v3 )
    {
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 )
        j_j___libc_free_0(v4, *(_QWORD *)(v3 + 24) - v4);
      j_j___libc_free_0(v3, 88);
    }
    j_j___libc_free_0(v2, 88);
  }
  return sub_BB9280((__int64)a1);
}
