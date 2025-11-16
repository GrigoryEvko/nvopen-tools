// Function: sub_31B0A30
// Address: 0x31b0a30
//
__int64 __fastcall sub_31B0A30(__int64 a1)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rdi

  if ( !*(_BYTE *)(a1 + 64) )
  {
    if ( !*(_BYTE *)(a1 + 80) )
      goto LABEL_3;
LABEL_24:
    sub_3187250(*(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 72));
    if ( !*(_BYTE *)(a1 + 96) )
      goto LABEL_4;
    goto LABEL_25;
  }
  sub_3187270(*(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 56));
  if ( *(_BYTE *)(a1 + 80) )
    goto LABEL_24;
LABEL_3:
  if ( !*(_BYTE *)(a1 + 96) )
    goto LABEL_4;
LABEL_25:
  sub_3187290(*(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 88));
LABEL_4:
  if ( *(_BYTE *)(a1 + 112) )
    sub_31875B0(*(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 104));
  v2 = *(_QWORD *)(a1 + 120);
  if ( v2 )
  {
    *(_QWORD *)(v2 + 528) = &unk_49DDBE8;
    if ( (*(_BYTE *)(v2 + 544) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v2 + 552), 16LL * *(unsigned int *)(v2 + 560), 8);
    nullsub_184();
    v3 = *(_QWORD *)(v2 + 376);
    if ( v3 != v2 + 392 )
      _libc_free(v3);
    if ( (*(_BYTE *)(v2 + 24) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v2 + 32), 40LL * *(unsigned int *)(v2 + 40), 8);
    j_j___libc_free_0(v2);
  }
  v4 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD **)(a1 + 8);
    v6 = &v5[2 * v4];
    do
    {
      if ( *v5 != -8192 && *v5 != -4096 )
      {
        v7 = v5[1];
        if ( v7 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
      }
      v5 += 2;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16 * v4, 8);
}
