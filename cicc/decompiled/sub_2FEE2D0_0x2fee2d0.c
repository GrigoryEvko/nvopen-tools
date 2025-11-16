// Function: sub_2FEE2D0
// Address: 0x2fee2d0
//
__int64 __fastcall sub_2FEE2D0(_QWORD *a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 v3; // rdi

  v1 = a1[33];
  *a1 = &unk_4A2D670;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 32);
    if ( v3 != v1 + 48 )
      _libc_free(v3);
    sub_C7D6A0(*(_QWORD *)(v1 + 8), 24LL * *(unsigned int *)(v1 + 24), 8);
    j_j___libc_free_0(v1);
  }
  return sub_BB9280((__int64)a1);
}
