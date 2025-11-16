// Function: sub_254B530
// Address: 0x254b530
//
void __fastcall sub_254B530(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rsi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  v1 = a1 - 88;
  v2 = *(unsigned int *)(a1 + 296);
  *(_QWORD *)(a1 - 88) = off_4A19BF0;
  *(_QWORD *)a1 = &unk_4A19CB0;
  sub_C7D6A0(*(_QWORD *)(a1 + 280), 8 * v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 248), 16LL * *(unsigned int *)(a1 + 264), 8);
  v4 = *(_QWORD *)(a1 + 160);
  if ( v4 != a1 + 176 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 8LL * *(unsigned int *)(a1 + 152), 8);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v5 != a1 + 64 )
    _libc_free(v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
  v6 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v6 != a1 - 32 )
    _libc_free(v6);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
