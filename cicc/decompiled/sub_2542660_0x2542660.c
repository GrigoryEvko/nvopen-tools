// Function: sub_2542660
// Address: 0x2542660
//
void __fastcall sub_2542660(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi

  v1 = a1 - 88;
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  v3 = *(_QWORD *)(a1 - 48);
  if ( v3 != a1 - 32 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
