// Function: sub_30D30A0
// Address: 0x30d30a0
//
__int64 __fastcall sub_30D30A0(__int64 a1)
{
  _QWORD *v2; // rdi
  bool v3; // zf
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned __int64 v7; // rdi
  bool v9; // cc
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi

  v2 = (_QWORD *)(a1 + 824);
  *(v2 - 103) = off_49D8928;
  *v2 = off_4A325F8;
  nullsub_35();
  sub_C7D6A0(*(_QWORD *)(a1 + 800), 16LL * *(unsigned int *)(a1 + 816), 8);
  if ( *(_BYTE *)(a1 + 768) )
  {
    v9 = *(_DWORD *)(a1 + 760) <= 0x40u;
    *(_BYTE *)(a1 + 768) = 0;
    if ( !v9 )
    {
      v10 = *(_QWORD *)(a1 + 752);
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    if ( *(_DWORD *)(a1 + 744) > 0x40u )
    {
      v11 = *(_QWORD *)(a1 + 736);
      if ( v11 )
        j_j___libc_free_0_0(v11);
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 680), 24LL * *(unsigned int *)(a1 + 696), 8);
  v3 = *(_BYTE *)(a1 + 492) == 0;
  *(_QWORD *)a1 = off_49D8850;
  if ( v3 )
    _libc_free(*(_QWORD *)(a1 + 472));
  sub_C7D6A0(*(_QWORD *)(a1 + 432), 16LL * *(unsigned int *)(a1 + 448), 8);
  if ( !*(_BYTE *)(a1 + 292) )
    _libc_free(*(_QWORD *)(a1 + 272));
  v4 = *(unsigned int *)(a1 + 256);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 240);
    v6 = v5 + 32 * v4;
    do
    {
      if ( *(_QWORD *)v5 != -8192 && *(_QWORD *)v5 != -4096 && *(_DWORD *)(v5 + 24) > 0x40u )
      {
        v7 = *(_QWORD *)(v5 + 16);
        if ( v7 )
          j_j___libc_free_0_0(v7);
      }
      v5 += 32;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 256);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 32 * v4, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 208), 8LL * *(unsigned int *)(a1 + 224), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 16LL * *(unsigned int *)(a1 + 192), 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 144), 16LL * *(unsigned int *)(a1 + 160), 8);
}
