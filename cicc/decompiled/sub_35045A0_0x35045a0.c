// Function: sub_35045A0
// Address: 0x35045a0
//
unsigned __int64 __fastcall sub_35045A0(unsigned __int64 *a1, unsigned __int64 **a2, __int64 a3)
{
  _QWORD *v5; // rax
  __int64 v6; // r9
  unsigned __int64 v7; // r12
  unsigned __int64 *v8; // rsi
  _BYTE *v9; // rdx
  _QWORD *v10; // r14
  _QWORD *v11; // r15
  __int64 *v12; // rcx
  size_t *v13; // rax
  unsigned __int64 v14; // r13
  __int64 v15; // rcx
  size_t v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // r9
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  _QWORD *v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // r13
  unsigned __int64 v25; // rdi
  __int64 v27; // rsi
  char v28; // al
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r14
  _QWORD *v31; // r15
  __int64 v32; // r8
  unsigned __int64 **v33; // rax
  unsigned __int64 *v34; // rdx
  _QWORD *v35; // r9
  _QWORD *v36; // rsi
  unsigned __int64 v37; // rdi
  _QWORD *v38; // rcx
  unsigned __int64 v39; // rdx
  _QWORD **v40; // rax
  unsigned __int64 v41; // rdx
  size_t n; // [rsp+8h] [rbp-38h]
  __int64 nb; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]

  v5 = (_QWORD *)sub_22077B0(0xC8u);
  v7 = (unsigned __int64)v5;
  if ( v5 )
    *v5 = 0;
  v8 = *a2;
  v9 = *(_BYTE **)a3;
  v10 = v5 + 8;
  v11 = v5 + 14;
  v12 = *(__int64 **)(a3 + 16);
  v13 = *(size_t **)(a3 + 24);
  v14 = *v8;
  *(_QWORD *)(v7 + 8) = *v8;
  LOBYTE(v9) = *v9;
  v15 = *v12;
  v16 = *v13;
  *(_QWORD *)(v7 + 48) = v7 + 64;
  *(_BYTE *)(v7 + 40) = (_BYTE)v9;
  *(_QWORD *)(v7 + 16) = v16;
  *(_QWORD *)(v7 + 24) = v15;
  *(_QWORD *)(v7 + 32) = 0;
  *(_QWORD *)(v7 + 56) = 0x400000000LL;
  *(_QWORD *)(v7 + 96) = v7 + 112;
  *(_QWORD *)(v7 + 104) = 0x400000000LL;
  *(_QWORD *)(v7 + 176) = 0;
  *(_QWORD *)(v7 + 184) = 0;
  *(_QWORD *)(v7 + 192) = 0;
  if ( v16 )
  {
    v17 = *(unsigned int *)(v16 + 40);
    if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 44) )
    {
      na = v16;
      sub_C8D5F0(v16 + 32, (const void *)(v16 + 48), v17 + 1, 8u, v17 + 1, v6);
      v16 = na;
      v17 = *(unsigned int *)(na + 40);
    }
    *(_QWORD *)(*(_QWORD *)(v16 + 32) + 8 * v17) = v7 + 16;
    ++*(_DWORD *)(v16 + 40);
    v14 = *(_QWORD *)(v7 + 8);
  }
  v18 = a1[1];
  v19 = *(_QWORD **)(*a1 + 8 * (v14 % v18));
  if ( !v19 )
    goto LABEL_19;
  v20 = (_QWORD *)*v19;
  if ( *(_QWORD *)(*v19 + 8LL) != v14 )
  {
    do
    {
      v21 = (_QWORD *)*v20;
      if ( !*v20 )
        goto LABEL_19;
      v19 = v20;
      if ( v14 % v18 != v21[1] % v18 )
        goto LABEL_19;
      v20 = (_QWORD *)*v20;
    }
    while ( v21[1] != v14 );
  }
  v22 = *v19;
  if ( !*v19 )
  {
LABEL_19:
    v27 = a1[1];
    n = 8 * (v14 % v18);
    v28 = sub_222DA10((__int64)(a1 + 4), v27, a1[3], 1);
    v30 = v29;
    if ( !v28 )
    {
      v31 = (_QWORD *)*a1;
      v32 = n;
      v33 = (unsigned __int64 **)(*a1 + n);
      v34 = *v33;
      if ( *v33 )
      {
LABEL_21:
        *(_QWORD *)v7 = *v34;
        **v33 = v7;
LABEL_22:
        ++a1[3];
        return v7;
      }
LABEL_36:
      v41 = a1[2];
      a1[2] = v7;
      *(_QWORD *)v7 = v41;
      if ( v41 )
      {
        v31[*(_QWORD *)(v41 + 8) % a1[1]] = v7;
        v33 = (unsigned __int64 **)(v32 + *a1);
      }
      *v33 = a1 + 2;
      goto LABEL_22;
    }
    if ( v29 == 1 )
    {
      v31 = a1 + 6;
      a1[6] = 0;
      v35 = a1 + 6;
    }
    else
    {
      if ( v29 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v27, v29);
      nb = 8 * v29;
      v31 = (_QWORD *)sub_22077B0(8 * v29);
      memset(v31, 0, nb);
      v35 = a1 + 6;
    }
    v36 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v36 )
    {
LABEL_33:
      if ( v35 != (_QWORD *)*a1 )
        j_j___libc_free_0(*a1);
      a1[1] = v30;
      *a1 = (unsigned __int64)v31;
      v32 = 8 * (v14 % v30);
      v33 = (unsigned __int64 **)((char *)v31 + v32);
      v34 = *(unsigned __int64 **)((char *)v31 + v32);
      if ( v34 )
        goto LABEL_21;
      goto LABEL_36;
    }
    v37 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v38 = v36;
        v36 = (_QWORD *)*v36;
        v39 = v38[1] % v30;
        v40 = (_QWORD **)&v31[v39];
        if ( !*v40 )
          break;
        *v38 = **v40;
        **v40 = v38;
LABEL_29:
        if ( !v36 )
          goto LABEL_33;
      }
      *v38 = a1[2];
      a1[2] = (unsigned __int64)v38;
      *v40 = a1 + 2;
      if ( !*v38 )
      {
        v37 = v39;
        goto LABEL_29;
      }
      v31[v37] = v38;
      v37 = v39;
      if ( !v36 )
        goto LABEL_33;
    }
  }
  v23 = *(_QWORD *)(v7 + 96);
  v24 = v22;
  if ( v11 != (_QWORD *)v23 )
    _libc_free(v23);
  v25 = *(_QWORD *)(v7 + 48);
  if ( v10 != (_QWORD *)v25 )
    _libc_free(v25);
  j_j___libc_free_0(v7);
  return v24;
}
