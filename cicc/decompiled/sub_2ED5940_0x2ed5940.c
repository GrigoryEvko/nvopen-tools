// Function: sub_2ED5940
// Address: 0x2ed5940
//
void __fastcall sub_2ED5940(__int64 a1)
{
  __int64 v1; // rsi
  _QWORD *v3; // r12
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // r12
  __int64 i; // rbx
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  unsigned __int64 *v26; // rax
  unsigned __int64 v27; // r13

  v1 = *(unsigned int *)(a1 + 1240);
  if ( (_DWORD)v1 )
  {
    v3 = *(_QWORD **)(a1 + 1224);
    v4 = &v3[4 * v1];
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v5 = v3[1];
        if ( v5 )
          j_j___libc_free_0(v5);
      }
      v3 += 4;
    }
    while ( v4 != v3 );
    v1 = *(unsigned int *)(a1 + 1240);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1224), 32 * v1, 8);
  v6 = *(unsigned int *)(a1 + 1208);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD **)(a1 + 1192);
    v8 = &v7[10 * v6];
    while ( 1 )
    {
      while ( *v7 == -4096 )
      {
        if ( v7[1] != -4096 )
          goto LABEL_12;
        v7 += 10;
        if ( v8 == v7 )
        {
LABEL_18:
          v6 = *(unsigned int *)(a1 + 1208);
          goto LABEL_19;
        }
      }
      if ( *v7 != -8192 || v7[1] != -8192 )
      {
LABEL_12:
        v9 = v7[2];
        if ( (_QWORD *)v9 != v7 + 4 )
          _libc_free(v9);
      }
      v7 += 10;
      if ( v8 == v7 )
        goto LABEL_18;
    }
  }
LABEL_19:
  sub_C7D6A0(*(_QWORD *)(a1 + 1192), 80 * v6, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1160), 24LL * *(unsigned int *)(a1 + 1176), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1128), 40LL * *(unsigned int *)(a1 + 1144), 8);
  if ( (*(_BYTE *)(a1 + 1048) & 1) != 0 )
  {
    v10 = a1 + 1056;
    v13 = a1 + 1120;
    goto LABEL_54;
  }
  v10 = *(_QWORD *)(a1 + 1056);
  v11 = *(unsigned int *)(a1 + 1064);
  v12 = 16 * v11;
  if ( (_DWORD)v11 && (v13 = v10 + v12, v10 != v10 + v12) )
  {
    do
    {
LABEL_54:
      while ( *(_DWORD *)v10 > 0xFFFFFFFD )
      {
        v10 += 16;
        if ( v10 == v13 )
          goto LABEL_56;
      }
      v25 = *(_QWORD *)(v10 + 8);
      if ( v25 )
      {
        if ( (v25 & 2) != 0 )
        {
          v26 = (unsigned __int64 *)(v25 & 0xFFFFFFFFFFFFFFFCLL);
          v27 = (unsigned __int64)v26;
          if ( v26 )
          {
            if ( (unsigned __int64 *)*v26 != v26 + 2 )
              _libc_free(*v26);
            j_j___libc_free_0(v27);
          }
        }
      }
      v10 += 16;
    }
    while ( v10 != v13 );
LABEL_56:
    if ( (*(_BYTE *)(a1 + 1048) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(a1 + 1056), 16LL * *(unsigned int *)(a1 + 1064), 8);
  }
  else
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 1056), v12, 8);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1016), 4LL * *(unsigned int *)(a1 + 1032), 4);
  v14 = *(_QWORD *)(a1 + 992);
  if ( v14 != a1 + 1008 )
    _libc_free(v14);
  sub_C7D6A0(*(_QWORD *)(a1 + 968), 16LL * *(unsigned int *)(a1 + 984), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 936), 24LL * *(unsigned int *)(a1 + 952), 8);
  sub_2ED2150(*(_QWORD *)(a1 + 896));
  v15 = *(_QWORD *)(a1 + 736);
  if ( v15 != a1 + 752 )
    _libc_free(v15);
  v16 = *(_QWORD *)(a1 + 616);
  if ( v16 != a1 + 632 )
    _libc_free(v16);
  v17 = *(_QWORD *)(a1 + 384);
  if ( v17 )
    j_j___libc_free_0_0(v17);
  v18 = *(_QWORD *)(a1 + 312);
  if ( v18 != a1 + 328 )
    _libc_free(v18);
  v19 = *(_QWORD *)(a1 + 240);
  if ( v19 != a1 + 256 )
    _libc_free(v19);
  v20 = *(_QWORD *)(a1 + 176);
  if ( v20 != a1 + 200 )
    _libc_free(v20);
  v21 = *(_QWORD *)(a1 + 120);
  if ( v21 != a1 + 144 )
    _libc_free(v21);
  v22 = *(_QWORD *)(a1 + 88);
  if ( v22 )
  {
    for ( i = v22 + 24LL * *(_QWORD *)(v22 - 8); v22 != i; i -= 24 )
    {
      v24 = *(_QWORD *)(i - 8);
      if ( v24 )
        j_j___libc_free_0_0(v24);
    }
    j_j_j___libc_free_0_0(v22 - 8);
  }
}
