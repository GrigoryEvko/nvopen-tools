// Function: sub_2546160
// Address: 0x2546160
//
void __fastcall sub_2546160(__int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1 - 88;
  *(_QWORD *)(a1 - 88) = off_4A19650;
  *(_QWORD *)a1 = &unk_4A19710;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != a1 + 64 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
  v4 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v4 != a1 - 32 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
