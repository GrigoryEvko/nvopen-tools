// Function: sub_25DE530
// Address: 0x25de530
//
void __fastcall sub_25DE530(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *i; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  unsigned __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 *v17; // r13
  __int64 v18; // r15
  __int64 v19; // rbx
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  unsigned __int64 *v27; // rbx
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 j; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+18h] [rbp-48h]
  __int64 v35; // [rsp+20h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 160);
  for ( i = &v2[*(unsigned int *)(a1 + 168)]; i != v2; ++v2 )
  {
    v4 = *v2;
    if ( *(_QWORD *)(*v2 + 16) )
    {
      a2 = sub_AD6530(*(_QWORD *)(v4 + 8), a2);
      sub_BD84D0(v4, a2);
    }
  }
  if ( *(_BYTE *)(a1 + 556) )
  {
    v5 = a1;
    if ( *(_BYTE *)(a1 + 460) )
      goto LABEL_7;
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 536));
    v5 = a1;
    if ( *(_BYTE *)(a1 + 460) )
      goto LABEL_7;
  }
  _libc_free(*(_QWORD *)(v5 + 440));
LABEL_7:
  v6 = *(_QWORD *)(a1 + 160);
  v7 = v6 + 8LL * *(unsigned int *)(a1 + 168);
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v7 - 8);
      v7 -= 8LL;
      if ( v8 )
      {
        sub_B30220(v8);
        *(_DWORD *)(v8 + 4) = *(_DWORD *)(v8 + 4) & 0xF8000000 | 1;
        sub_B2F9E0(v8, a2, v9, v10);
        sub_BD2DD0(v8);
      }
    }
    while ( v6 != v7 );
    v7 = *(_QWORD *)(a1 + 160);
  }
  if ( v7 != a1 + 176 )
    _libc_free(v7);
  v11 = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)v11 )
  {
    v12 = *(_QWORD **)(a1 + 136);
    v13 = &v12[2 * v11];
    do
    {
      if ( *v12 != -8192 && *v12 != -4096 )
        sub_29CF750(v12 + 1);
      v12 += 2;
    }
    while ( v13 != v12 );
    LODWORD(v11) = *(_DWORD *)(a1 + 152);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 16LL * (unsigned int)v11, 8);
  v14 = *(_QWORD *)(a1 + 80);
  if ( v14 != a1 + 96 )
    _libc_free(v14);
  v15 = *(_QWORD *)(a1 + 56);
  v16 = *(_QWORD *)(a1 + 16);
  v35 = *(_QWORD *)(a1 + 48);
  v33 = *(_QWORD *)(a1 + 32);
  v34 = *(_QWORD *)(a1 + 72);
  v17 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL);
  for ( j = *(_QWORD *)(a1 + 40); v34 > (unsigned __int64)v17; ++v17 )
  {
    v18 = *v17;
    v19 = *v17 + 512;
    do
    {
      v20 = *(unsigned int *)(v18 + 24);
      v21 = *(_QWORD *)(v18 + 8);
      v18 += 32;
      sub_C7D6A0(v21, 16 * v20, 8);
    }
    while ( v19 != v18 );
  }
  if ( v34 == j )
  {
    while ( v35 != v16 )
    {
      v30 = *(unsigned int *)(v16 + 24);
      v31 = *(_QWORD *)(v16 + 8);
      v16 += 32;
      sub_C7D6A0(v31, 16 * v30, 8);
    }
    v26 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 )
      return;
    goto LABEL_33;
  }
  while ( v33 != v16 )
  {
    v22 = *(unsigned int *)(v16 + 24);
    v23 = *(_QWORD *)(v16 + 8);
    v16 += 32;
    sub_C7D6A0(v23, 16 * v22, 8);
  }
  while ( v35 != v15 )
  {
    v24 = *(unsigned int *)(v15 + 24);
    v25 = *(_QWORD *)(v15 + 8);
    v15 += 32;
    sub_C7D6A0(v25, 16 * v24, 8);
  }
  v26 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
LABEL_33:
    v27 = *(unsigned __int64 **)(a1 + 40);
    v28 = *(_QWORD *)(a1 + 72) + 8LL;
    if ( v28 > (unsigned __int64)v27 )
    {
      do
      {
        v29 = *v27++;
        j_j___libc_free_0(v29);
      }
      while ( v28 > (unsigned __int64)v27 );
      v26 = *(_QWORD *)a1;
    }
    j_j___libc_free_0(v26);
  }
}
