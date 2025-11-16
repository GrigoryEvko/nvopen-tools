// Function: sub_2EF2DE0
// Address: 0x2ef2de0
//
void __fastcall sub_2EF2DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r12
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi

  v7 = *(_QWORD *)(a1 + 696);
  v8 = v7 + 8LL * *(unsigned int *)(a1 + 704);
  if ( v7 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v8 - 8);
      v8 -= 8LL;
      if ( v9 )
      {
        v10 = *(_QWORD *)(v9 + 24);
        if ( v10 != v9 + 40 )
          _libc_free(v10);
        a2 = 80;
        j_j___libc_free_0(v9);
      }
    }
    while ( v7 != v8 );
    v8 = *(_QWORD *)(a1 + 696);
  }
  if ( v8 != a1 + 712 )
    _libc_free(v8);
  v11 = *(_QWORD *)(a1 + 672);
  if ( v11 != a1 + 688 )
    _libc_free(v11);
  sub_2EF2CF0(a1 + 664, a2, a3, a4, a5, a6);
  v12 = *(unsigned int *)(a1 + 624);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD *)(a1 + 608);
    v14 = v13 + 368LL * (unsigned int)v12;
    while ( 1 )
    {
      if ( *(_QWORD *)v13 == -8192 || *(_QWORD *)v13 == -4096 )
        goto LABEL_16;
      if ( *(_BYTE *)(v13 + 300) )
      {
        if ( !*(_BYTE *)(v13 + 204) )
          goto LABEL_21;
      }
      else
      {
        _libc_free(*(_QWORD *)(v13 + 280));
        if ( !*(_BYTE *)(v13 + 204) )
LABEL_21:
          _libc_free(*(_QWORD *)(v13 + 184));
      }
      sub_C7D6A0(*(_QWORD *)(v13 + 152), 4LL * *(unsigned int *)(v13 + 168), 4);
      sub_C7D6A0(*(_QWORD *)(v13 + 120), 4LL * *(unsigned int *)(v13 + 136), 4);
      sub_C7D6A0(*(_QWORD *)(v13 + 88), 4LL * *(unsigned int *)(v13 + 104), 4);
      sub_C7D6A0(*(_QWORD *)(v13 + 56), 4LL * *(unsigned int *)(v13 + 72), 4);
      sub_C7D6A0(*(_QWORD *)(v13 + 24), 16LL * *(unsigned int *)(v13 + 40), 8);
LABEL_16:
      v13 += 368;
      if ( v14 == v13 )
      {
        v12 = *(unsigned int *)(a1 + 624);
        break;
      }
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 608), 368 * v12, 8);
  v15 = *(_QWORD *)(a1 + 544);
  if ( v15 != a1 + 560 )
    _libc_free(v15);
  v16 = *(_QWORD *)(a1 + 464);
  if ( v16 != a1 + 480 )
    _libc_free(v16);
  v17 = *(_QWORD *)(a1 + 384);
  if ( v17 != a1 + 400 )
    _libc_free(v17);
  v18 = *(_QWORD *)(a1 + 304);
  if ( v18 != a1 + 320 )
    _libc_free(v18);
  sub_C7D6A0(*(_QWORD *)(a1 + 280), 4LL * *(unsigned int *)(a1 + 296), 4);
  v19 = *(_QWORD *)(a1 + 200);
  if ( v19 != a1 + 216 )
    _libc_free(v19);
  if ( !*(_BYTE *)(a1 + 132) )
    _libc_free(*(_QWORD *)(a1 + 112));
}
