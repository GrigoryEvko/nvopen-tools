// Function: sub_25466A0
// Address: 0x25466a0
//
void __fastcall sub_25466A0(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1 - 88;
  *(_QWORD *)(a1 - 88) = &unk_4A171F8;
  *(_QWORD *)a1 = &unk_4A171B8;
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 != a1 + 72 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 24LL * *(unsigned int *)(a1 + 48), 8);
  v4 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v4 != a1 - 32 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
