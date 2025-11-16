// Function: sub_2546BC0
// Address: 0x2546bc0
//
void __fastcall sub_2546BC0(unsigned __int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 168);
  *(_QWORD *)a1 = &unk_4A17358;
  *(_QWORD *)(a1 + 88) = &unk_4A17318;
  sub_C7D6A0(*(_QWORD *)(a1 + 152), v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 16LL * *(unsigned int *)(a1 + 128), 8);
  v3 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v3 != a1 + 56 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
