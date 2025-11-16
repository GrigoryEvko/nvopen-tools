// Function: sub_3424860
// Address: 0x3424860
//
void __fastcall sub_3424860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi

  v7 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)a1 = &unk_4A36970;
  if ( v7 )
  {
    sub_33CC6B0(v7, a2, a3, a4, a5, a6);
    j_j___libc_free_0(v7);
  }
  v8 = *(_QWORD *)(a1 + 32);
  if ( v8 )
  {
    v9 = *(_QWORD *)(v8 + 136);
    if ( v9 != v8 + 152 )
      _libc_free(v9);
    sub_C7D6A0(*(_QWORD *)(v8 + 104), 16LL * *(unsigned int *)(v8 + 120), 8);
    sub_C7D6A0(*(_QWORD *)(v8 + 72), 24LL * *(unsigned int *)(v8 + 88), 8);
    sub_C7D6A0(*(_QWORD *)(v8 + 40), 24LL * *(unsigned int *)(v8 + 56), 8);
    j_j___libc_free_0(v8);
  }
  v10 = *(_QWORD *)(a1 + 928);
  if ( v10 )
    j_j___libc_free_0(v10);
  v11 = *(_QWORD *)(a1 + 888);
  if ( v11 )
  {
    v12 = *(_QWORD *)(v11 + 16);
    if ( v12 )
    {
      sub_FDC110(*(__int64 **)(v11 + 16));
      j_j___libc_free_0(v12);
    }
    j_j___libc_free_0(v11);
  }
  if ( *(_BYTE *)(a1 + 852) )
  {
    if ( !*(_BYTE *)(a1 + 760) )
      goto LABEL_15;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 832));
    if ( !*(_BYTE *)(a1 + 760) )
      goto LABEL_15;
  }
  *(_BYTE *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 608) = &unk_49DDBE8;
  if ( (*(_BYTE *)(a1 + 624) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 632), 16LL * *(unsigned int *)(a1 + 640), 8);
  nullsub_184();
  v15 = *(_QWORD *)(a1 + 456);
  if ( v15 != a1 + 472 )
    _libc_free(v15);
  if ( (*(_BYTE *)(a1 + 104) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 112), 40LL * *(unsigned int *)(a1 + 120), 8);
LABEL_15:
  v13 = *(_QWORD *)(a1 + 72);
  if ( v13 )
  {
    sub_3424320(*(_QWORD *)(a1 + 72));
    j_j___libc_free_0(v13);
  }
  v14 = *(_QWORD *)(a1 + 24);
  if ( v14 )
  {
    sub_3424570(*(_QWORD *)(a1 + 24));
    j_j___libc_free_0(v14);
  }
}
