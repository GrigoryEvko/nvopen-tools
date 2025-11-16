// Function: sub_2546DD0
// Address: 0x2546dd0
//
void __fastcall sub_2546DD0(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rsi
  unsigned __int64 v4; // rdi

  v1 = a1 - 88;
  v2 = *(unsigned int *)(a1 + 80);
  *(_QWORD *)(a1 - 88) = &unk_4A17358;
  *(_QWORD *)a1 = &unk_4A17318;
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 16 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
  v4 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v4 != a1 - 32 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
