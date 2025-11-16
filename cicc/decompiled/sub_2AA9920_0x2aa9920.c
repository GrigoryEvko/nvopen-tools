// Function: sub_2AA9920
// Address: 0x2aa9920
//
void __fastcall sub_2AA9920(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi

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
  v5 = a1 + 112;
  *(_QWORD *)(v5 + 112) = &unk_49DA100;
  nullsub_63();
  v6 = *(_QWORD *)(v5 - 16);
  if ( v6 != v5 )
    _libc_free(v6);
}
