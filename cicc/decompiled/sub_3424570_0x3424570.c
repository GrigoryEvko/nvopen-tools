// Function: sub_3424570
// Address: 0x3424570
//
void __fastcall sub_3424570(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // r13
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi

  if ( *(_DWORD *)(a1 + 1136) > 0x40u )
  {
    v2 = *(_QWORD *)(a1 + 1128);
    if ( v2 )
      j_j___libc_free_0_0(v2);
  }
  if ( *(_DWORD *)(a1 + 1120) > 0x40u )
  {
    v3 = *(_QWORD *)(a1 + 1112);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  v4 = *(_QWORD *)(a1 + 1088);
  v5 = v4 + 40LL * *(unsigned int *)(a1 + 1096);
  if ( v4 != v5 )
  {
    do
    {
      v5 -= 40LL;
      if ( *(_DWORD *)(v5 + 32) > 0x40u )
      {
        v6 = *(_QWORD *)(v5 + 24);
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      if ( *(_DWORD *)(v5 + 16) > 0x40u )
      {
        v7 = *(_QWORD *)(v5 + 8);
        if ( v7 )
          j_j___libc_free_0_0(v7);
      }
    }
    while ( v4 != v5 );
    v5 = *(_QWORD *)(a1 + 1088);
  }
  if ( v5 != a1 + 1104 )
    _libc_free(v5);
  if ( !*(_BYTE *)(a1 + 1020) )
    _libc_free(*(_QWORD *)(a1 + 1000));
  if ( !*(_BYTE *)(a1 + 924) )
    _libc_free(*(_QWORD *)(a1 + 904));
  v8 = *(_QWORD *)(a1 + 856);
  if ( v8 )
    j_j___libc_free_0(v8);
  v9 = *(_QWORD *)(a1 + 792);
  if ( v9 != a1 + 816 )
    _libc_free(v9);
  sub_C7D6A0(*(_QWORD *)(a1 + 768), 16LL * *(unsigned int *)(a1 + 784), 8);
  v10 = *(_QWORD *)(a1 + 528);
  if ( v10 != a1 + 544 )
    _libc_free(v10);
  sub_C7D6A0(*(_QWORD *)(a1 + 504), 4LL * *(unsigned int *)(a1 + 520), 4);
  sub_C7D6A0(*(_QWORD *)(a1 + 472), 8LL * *(unsigned int *)(a1 + 488), 4);
  v11 = *(_QWORD *)(a1 + 392);
  if ( v11 != a1 + 408 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 312);
  if ( v12 != a1 + 328 )
    _libc_free(v12);
  sub_C7D6A0(*(_QWORD *)(a1 + 288), 16LL * *(unsigned int *)(a1 + 304), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 256), 16LL * *(unsigned int *)(a1 + 272), 8);
  v13 = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)v13 )
  {
    v14 = *(_QWORD *)(a1 + 224);
    v15 = v14 + 40 * v13;
    do
    {
      if ( *(_QWORD *)v14 != -8192 && *(_QWORD *)v14 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v14 + 16), 16LL * *(unsigned int *)(v14 + 32), 8);
      v14 += 40;
    }
    while ( v15 != v14 );
    v13 = *(unsigned int *)(a1 + 240);
  }
  v16 = *(_QWORD *)(a1 + 224);
  v17 = a1 + 72;
  sub_C7D6A0(v16, 40 * v13, 8);
  sub_C7D6A0(*(_QWORD *)(v17 + 120), 16LL * *(unsigned int *)(v17 + 136), 8);
  sub_C7D6A0(*(_QWORD *)(v17 + 88), 16LL * *(unsigned int *)(v17 + 104), 8);
  sub_C7D6A0(*(_QWORD *)(v17 + 56), 16LL * *(unsigned int *)(v17 + 72), 8);
  v18 = *(_QWORD *)(v17 - 16);
  if ( v18 != v17 )
    _libc_free(v18);
}
