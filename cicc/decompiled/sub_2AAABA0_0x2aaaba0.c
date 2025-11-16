// Function: sub_2AAABA0
// Address: 0x2aaaba0
//
void __fastcall sub_2AAABA0(unsigned __int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 448);
  *(_QWORD *)a1 = &unk_4A23660;
  sub_C7D6A0(*(_QWORD *)(a1 + 432), v2, 8);
  v3 = *(_QWORD *)(a1 + 312);
  if ( v3 != a1 + 328 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 264);
  if ( v4 != a1 + 280 )
    _libc_free(v4);
  nullsub_61();
  *(_QWORD *)(a1 + 224) = &unk_49DA100;
  nullsub_63();
  v5 = *(_QWORD *)(a1 + 96);
  if ( v5 != a1 + 112 )
    _libc_free(v5);
  j_j___libc_free_0(a1);
}
