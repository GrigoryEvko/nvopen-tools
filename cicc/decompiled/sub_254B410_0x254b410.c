// Function: sub_254B410
// Address: 0x254b410
//
void __fastcall sub_254B410(unsigned __int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = off_4A19BF0;
  v2 = *(unsigned int *)(a1 + 384);
  *(_QWORD *)(a1 + 88) = &unk_4A19CB0;
  sub_C7D6A0(*(_QWORD *)(a1 + 368), 8 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 336), 16LL * *(unsigned int *)(a1 + 352), 8);
  v3 = *(_QWORD *)(a1 + 248);
  if ( v3 != a1 + 264 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 8LL * *(unsigned int *)(a1 + 240), 8);
  v4 = *(_QWORD *)(a1 + 136);
  if ( v4 != a1 + 152 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 8LL * *(unsigned int *)(a1 + 128), 8);
  v5 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v5 != a1 + 56 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
