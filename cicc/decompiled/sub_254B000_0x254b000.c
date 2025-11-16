// Function: sub_254B000
// Address: 0x254b000
//
__int64 __fastcall sub_254B000(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *(_QWORD *)(a1 - 88) = off_4A19BF0;
  v2 = *(unsigned int *)(a1 + 296);
  *(_QWORD *)a1 = &unk_4A19CB0;
  sub_C7D6A0(*(_QWORD *)(a1 + 280), 8 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 248), 16LL * *(unsigned int *)(a1 + 264), 8);
  v3 = *(_QWORD *)(a1 + 160);
  if ( v3 != a1 + 176 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 8LL * *(unsigned int *)(a1 + 152), 8);
  v4 = *(_QWORD *)(a1 + 48);
  if ( v4 != a1 + 64 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
  v5 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v5 != a1 - 32 )
    _libc_free(v5);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
