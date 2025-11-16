// Function: sub_2F43140
// Address: 0x2f43140
//
void __fastcall sub_2F43140(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  void (__fastcall *v19)(__int64, __int64, __int64); // rax
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // r12
  __int64 i; // rbx
  unsigned __int64 v27; // rdi

  sub_C7D6A0(*(_QWORD *)(a1 + 1264), 16LL * *(unsigned int *)(a1 + 1280), 8);
  v2 = *(_QWORD *)(a1 + 1176);
  if ( v2 != a1 + 1192 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 1128);
  if ( v3 != a1 + 1144 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 1112);
  if ( a1 + 1128 != v4 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 832);
  if ( v5 != a1 + 848 )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 808);
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD *)(a1 + 736);
  if ( v7 != a1 + 752 )
    _libc_free(v7);
  v8 = *(unsigned int *)(a1 + 728);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(a1 + 712);
    v10 = v9 + 32 * v8;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v9 <= 0xFFFFFFFD )
        {
          v11 = *(_QWORD *)(v9 + 8);
          if ( v11 != v9 + 24 )
            break;
        }
        v9 += 32;
        if ( v10 == v9 )
          goto LABEL_19;
      }
      _libc_free(v11);
      v9 += 32;
    }
    while ( v10 != v9 );
LABEL_19:
    v8 = *(unsigned int *)(a1 + 728);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 712), 32 * v8, 8);
  v12 = *(unsigned int *)(a1 + 696);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD *)(a1 + 680);
    v14 = v13 + 40 * v12;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v13 <= 0xFFFFFFFD )
        {
          v15 = *(_QWORD *)(v13 + 8);
          if ( v15 != v13 + 24 )
            break;
        }
        v13 += 40;
        if ( v14 == v13 )
          goto LABEL_26;
      }
      _libc_free(v15);
      v13 += 40;
    }
    while ( v14 != v13 );
LABEL_26:
    v12 = *(unsigned int *)(a1 + 696);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 680), 40 * v12, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 648), 32LL * *(unsigned int *)(a1 + 664), 8);
  v16 = *(_QWORD *)(a1 + 624);
  if ( v16 )
    _libc_free(v16);
  v17 = *(_QWORD *)(a1 + 416);
  if ( v17 != a1 + 432 )
    _libc_free(v17);
  v18 = *(_QWORD *)(a1 + 392);
  if ( v18 != a1 + 408 )
    _libc_free(v18);
  v19 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 368);
  if ( v19 )
    v19(a1 + 352, a1 + 352, 3);
  v20 = *(_QWORD *)(a1 + 328);
  if ( v20 )
    j_j___libc_free_0_0(v20);
  v21 = *(_QWORD *)(a1 + 256);
  if ( v21 != a1 + 272 )
    _libc_free(v21);
  v22 = *(_QWORD *)(a1 + 184);
  if ( v22 != a1 + 200 )
    _libc_free(v22);
  v23 = *(_QWORD *)(a1 + 120);
  if ( v23 != a1 + 144 )
    _libc_free(v23);
  v24 = *(_QWORD *)(a1 + 64);
  if ( v24 != a1 + 88 )
    _libc_free(v24);
  v25 = *(_QWORD *)(a1 + 32);
  if ( v25 )
  {
    for ( i = v25 + 24LL * *(_QWORD *)(v25 - 8); v25 != i; i -= 24 )
    {
      v27 = *(_QWORD *)(i - 8);
      if ( v27 )
        j_j___libc_free_0_0(v27);
    }
    j_j_j___libc_free_0_0(v25 - 8);
  }
}
