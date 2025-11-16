// Function: sub_2670DE0
// Address: 0x2670de0
//
void __fastcall sub_2670DE0(__int64 a1)
{
  unsigned __int64 v1; // r12
  bool v3; // zf
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v1 = a1 - 88;
  *(_QWORD *)(a1 - 88) = off_4A20228;
  v3 = *(_BYTE *)(a1 + 124) == 0;
  *(_QWORD *)a1 = &unk_4A202B8;
  if ( v3 )
    _libc_free(*(_QWORD *)(a1 + 104));
  v4 = *(_QWORD *)(a1 + 48);
  if ( v4 != a1 + 64 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
  v5 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v5 != a1 - 32 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
