// Function: sub_2546930
// Address: 0x2546930
//
void __fastcall sub_2546930(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1 - 104;
  *(_QWORD *)(a1 - 104) = &unk_4A1D118;
  *(_QWORD *)(a1 - 16) = &unk_4A1D1C0;
  *(_QWORD *)a1 = &unk_4A1D220;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != a1 + 64 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
  v4 = *(_QWORD *)(a1 - 64);
  *(_QWORD *)(a1 - 104) = &unk_4A16C00;
  if ( v4 != a1 - 48 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 - 88), 8LL * *(unsigned int *)(a1 - 72), 8);
  j_j___libc_free_0(v1);
}
