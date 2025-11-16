// Function: sub_2F61430
// Address: 0x2f61430
//
void __fastcall sub_2F61430(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // r12
  __int64 i; // rbx
  unsigned __int64 v21; // rdi

  *(_QWORD *)a1 = off_4A2B718;
  sub_C7D6A0(*(_QWORD *)(a1 + 928), 16LL * *(unsigned int *)(a1 + 944), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 896), 4LL * *(unsigned int *)(a1 + 912), 4);
  v2 = *(_QWORD *)(a1 + 840);
  if ( v2 != a1 + 856 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 760);
  if ( v3 != a1 + 776 )
    _libc_free(v3);
  if ( !*(_BYTE *)(a1 + 692) )
    _libc_free(*(_QWORD *)(a1 + 672));
  v4 = *(_QWORD *)(a1 + 584);
  if ( v4 != a1 + 600 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 504);
  if ( v5 != a1 + 520 )
    _libc_free(v5);
  v6 = *(unsigned int *)(a1 + 472);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(a1 + 456);
    v8 = v7 + 32 * v6;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v7 <= 0xFFFFFFFD )
        {
          v9 = *(_QWORD *)(v7 + 8);
          if ( v9 )
            break;
        }
        v7 += 32;
        if ( v8 == v7 )
          goto LABEL_17;
      }
      v7 += 32;
      j_j___libc_free_0(v9);
    }
    while ( v8 != v7 );
LABEL_17:
    v6 = *(unsigned int *)(a1 + 472);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 456), 32 * v6, 8);
  v10 = *(unsigned int *)(a1 + 440);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD *)(a1 + 424);
    v12 = v11 + 32 * v10;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v11 <= 0xFFFFFFFD )
        {
          v13 = *(_QWORD *)(v11 + 8);
          if ( v13 != v11 + 24 )
            break;
        }
        v11 += 32;
        if ( v12 == v11 )
          goto LABEL_24;
      }
      _libc_free(v13);
      v11 += 32;
    }
    while ( v12 != v11 );
LABEL_24:
    v10 = *(unsigned int *)(a1 + 440);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 424), 32 * v10, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 392), 24LL * *(unsigned int *)(a1 + 408), 8);
  v14 = *(_QWORD *)(a1 + 360);
  if ( v14 )
    j_j___libc_free_0_0(v14);
  v15 = *(_QWORD *)(a1 + 288);
  if ( v15 != a1 + 304 )
    _libc_free(v15);
  v16 = *(_QWORD *)(a1 + 216);
  if ( v16 != a1 + 232 )
    _libc_free(v16);
  v17 = *(_QWORD *)(a1 + 152);
  if ( v17 != a1 + 176 )
    _libc_free(v17);
  v18 = *(_QWORD *)(a1 + 96);
  if ( v18 != a1 + 120 )
    _libc_free(v18);
  v19 = *(_QWORD *)(a1 + 64);
  if ( v19 )
  {
    for ( i = v19 + 24LL * *(_QWORD *)(v19 - 8); v19 != i; i -= 24 )
    {
      v21 = *(_QWORD *)(i - 8);
      if ( v21 )
        j_j___libc_free_0_0(v21);
    }
    j_j_j___libc_free_0_0(v19 - 8);
  }
}
