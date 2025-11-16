// Function: sub_917130
// Address: 0x917130
//
__int64 __fastcall sub_917130(__int64 a1)
{
  __int64 v2; // rsi
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 *v9; // r12
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 v13; // rdi

  v2 = *(unsigned int *)(a1 + 520);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 504);
    v4 = &v3[4 * v2];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[1];
        if ( v5 )
          j_j___libc_free_0(v5, v3[3] - v5);
      }
      v3 += 4;
    }
    while ( v4 != v3 );
    v2 = *(unsigned int *)(a1 + 520);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 504), 32 * v2, 8);
  v6 = 16LL * *(unsigned int *)(a1 + 488);
  sub_C7D6A0(*(_QWORD *)(a1 + 472), v6, 8);
  v7 = *(_QWORD *)(a1 + 456);
  if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
    sub_BD60C0(a1 + 440);
  v8 = *(__int64 **)(a1 + 424);
  v9 = *(__int64 **)(a1 + 416);
  if ( v8 != v9 )
  {
    do
    {
      v10 = *v9;
      if ( *v9 )
      {
        v6 = v9[2] - v10;
        j_j___libc_free_0(v10, v6);
      }
      v9 += 3;
    }
    while ( v8 != v9 );
    v9 = *(__int64 **)(a1 + 416);
  }
  if ( v9 )
  {
    v6 = *(_QWORD *)(a1 + 432) - (_QWORD)v9;
    j_j___libc_free_0(v9, v6);
  }
  sub_909CC0(*(_QWORD *)(a1 + 384));
  v11 = *(_QWORD *)(a1 + 336);
  while ( v11 )
  {
    sub_909AF0(*(_QWORD *)(v11 + 24));
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 16);
    v6 = 48;
    j_j___libc_free_0(v12, 48);
  }
  sub_909CC0(*(_QWORD *)(a1 + 288));
  nullsub_61(a1 + 184);
  *(_QWORD *)(a1 + 176) = &unk_49DA100;
  nullsub_63(a1 + 176);
  v13 = *(_QWORD *)(a1 + 48);
  if ( v13 != a1 + 64 )
    _libc_free(v13, v6);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
