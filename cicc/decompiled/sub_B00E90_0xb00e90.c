// Function: sub_B00E90
// Address: 0xb00e90
//
__int64 __fastcall sub_B00E90(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // r13
  __int64 *v6; // rax
  __int64 v7; // rbx
  int v8; // r15d
  int v9; // r15d
  unsigned int v10; // eax
  __int64 **v11; // rcx
  __int64 *v12; // rbx
  size_t v13; // rcx
  __int64 *v14; // r15
  __int64 v15; // rbx
  int v16; // r13d
  __int64 v17; // r14
  int v18; // eax
  int v19; // r9d
  size_t v20; // rdx
  size_t v21; // rcx
  int v22; // r10d
  unsigned int v23; // r8d
  __int64 *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rbx
  unsigned int v27; // r13d
  __int64 v28; // r14
  int v29; // esi
  __int64 *v30; // rdi
  __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  unsigned int v36; // r13d
  __int64 v37; // r14
  unsigned int v38; // eax
  __int64 *v39; // rcx
  __int64 v40; // rsi
  __int64 v42; // rsi
  __int64 v43; // rdi
  int v44; // eax
  int v45; // ecx
  int v46; // edi
  int v47; // r8d
  int v48; // eax
  __int64 *v49; // r9
  size_t v50; // [rsp+0h] [rbp-60h]
  int v51; // [rsp+8h] [rbp-58h]
  unsigned int v52; // [rsp+Ch] [rbp-54h]
  size_t v53; // [rsp+10h] [rbp-50h]
  int v54; // [rsp+10h] [rbp-50h]
  size_t n; // [rsp+18h] [rbp-48h]
  size_t nb; // [rsp+18h] [rbp-48h]
  size_t na; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v59; // [rsp+28h] [rbp-38h] BYREF

  v4 = a2;
  sub_AF50C0(a1);
  v6 = *(__int64 **)(a1 + 8);
  v7 = *v6;
  v8 = *(_DWORD *)(*v6 + 656);
  n = *(_QWORD *)(*v6 + 640);
  if ( !v8 )
  {
    v12 = *(__int64 **)(a1 + 136);
    v14 = &v12[*(unsigned int *)(a1 + 144)];
    if ( v14 == v12 )
      goto LABEL_18;
    while ( 1 )
    {
LABEL_7:
      while ( v4 != v12 )
      {
LABEL_6:
        if ( v14 == ++v12 )
          goto LABEL_10;
      }
      if ( !a3 )
      {
        v33 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)(*v4 + 136) + 8LL));
        *v4 = sub_B98A20(v33, a2, v34, v35);
        goto LABEL_6;
      }
      ++v12;
      *v4 = a3;
      if ( v14 == v12 )
      {
LABEL_10:
        v14 = *(__int64 **)(a1 + 136);
        v13 = *(unsigned int *)(a1 + 144);
        goto LABEL_11;
      }
    }
  }
  v9 = v8 - 1;
  v10 = v9 & sub_AF6940(*(__int64 **)(a1 + 136), *(_QWORD *)(a1 + 136) + 8LL * *(unsigned int *)(a1 + 144));
  v11 = (__int64 **)(n + 8LL * v10);
  a2 = *v11;
  if ( (__int64 *)a1 == *v11 )
  {
LABEL_3:
    *v11 = (__int64 *)-8192LL;
    --*(_DWORD *)(v7 + 648);
    ++*(_DWORD *)(v7 + 652);
  }
  else
  {
    v45 = 1;
    while ( a2 != (__int64 *)-4096LL )
    {
      v46 = v45 + 1;
      v10 = v9 & (v45 + v10);
      v11 = (__int64 **)(n + 8LL * v10);
      a2 = *v11;
      if ( (__int64 *)a1 == *v11 )
        goto LABEL_3;
      v45 = v46;
    }
  }
  v12 = *(__int64 **)(a1 + 136);
  v13 = *(unsigned int *)(a1 + 144);
  v14 = &v12[v13];
  if ( v12 != v14 )
    goto LABEL_7;
LABEL_11:
  v6 = *(__int64 **)(a1 + 8);
  v15 = *v6;
  v16 = *(_DWORD *)(*v6 + 656);
  v17 = *(_QWORD *)(*v6 + 640);
  if ( !v16 )
  {
LABEL_18:
    v26 = *v6;
    v58 = a1;
    v27 = *(_DWORD *)(v26 + 656);
    if ( v27 )
    {
      v36 = v27 - 1;
      v37 = *(_QWORD *)(v26 + 640);
      v38 = v36 & sub_AF6940(*(__int64 **)(a1 + 136), *(_QWORD *)(a1 + 136) + 8LL * *(unsigned int *)(a1 + 144));
      v39 = (__int64 *)(v37 + 8LL * v38);
      v31 = v58;
      v40 = *v39;
      if ( *v39 == v58 )
        return sub_AF5060(a1);
      v47 = 1;
      v30 = 0;
      while ( v40 != -4096 )
      {
        if ( v40 != -8192 || v30 )
          v39 = v30;
        v38 = v36 & (v47 + v38);
        v49 = (__int64 *)(v37 + 8LL * v38);
        v40 = *v49;
        if ( *v49 == v58 )
          return sub_AF5060(a1);
        ++v47;
        v30 = v39;
        v39 = (__int64 *)(v37 + 8LL * v38);
      }
      v48 = *(_DWORD *)(v26 + 648);
      v27 = *(_DWORD *)(v26 + 656);
      v28 = v26 + 632;
      if ( !v30 )
        v30 = v39;
      ++*(_QWORD *)(v26 + 632);
      v32 = v48 + 1;
      v59 = v30;
      if ( 4 * v32 < 3 * v27 )
      {
        if ( v27 - (v32 + *(_DWORD *)(v26 + 652)) > v27 >> 3 )
          goto LABEL_47;
        v29 = v27;
LABEL_21:
        sub_B00970(v28, v29);
        sub_AFC070(v28, &v58, &v59);
        v30 = v59;
        v31 = v58;
        v32 = *(_DWORD *)(v26 + 648) + 1;
LABEL_47:
        *(_DWORD *)(v26 + 648) = v32;
        if ( *v30 != -4096 )
          --*(_DWORD *)(v26 + 652);
        *v30 = v31;
        return sub_AF5060(a1);
      }
    }
    else
    {
      ++*(_QWORD *)(v26 + 632);
      v28 = v26 + 632;
      v59 = 0;
    }
    v29 = 2 * v27;
    goto LABEL_21;
  }
  v53 = v13;
  nb = 8 * v13;
  v18 = sub_AF6940(v14, (__int64)&v14[v13]);
  v19 = v16 - 1;
  v20 = nb;
  v21 = v53;
  v22 = 1;
  v23 = (v16 - 1) & v18;
  v24 = (__int64 *)(v17 + 8LL * v23);
  v25 = *v24;
  if ( *v24 == -4096 )
    goto LABEL_17;
  while ( 1 )
  {
    if ( v25 != -8192 && v21 == *(_DWORD *)(v25 + 144) )
    {
      v51 = v22;
      v52 = v23;
      v54 = v19;
      na = v21;
      if ( !v20 )
        break;
      v50 = v20;
      v44 = memcmp(v14, *(const void **)(v25 + 136), v20);
      v20 = v50;
      v21 = na;
      v19 = v54;
      v23 = v52;
      v22 = v51;
      if ( !v44 )
        break;
    }
    v23 = v19 & (v22 + v23);
    v24 = (__int64 *)(v17 + 8LL * v23);
    v25 = *v24;
    if ( *v24 == -4096 )
      goto LABEL_17;
    ++v22;
  }
  if ( v24 == (__int64 *)(*(_QWORD *)(v15 + 640) + 8LL * *(unsigned int *)(v15 + 656)) || (v42 = *v24) == 0 )
  {
LABEL_17:
    v6 = *(__int64 **)(a1 + 8);
    goto LABEL_18;
  }
  sub_BA6110(a1 + 8, v42);
  *(_DWORD *)(a1 + 144) = 0;
  sub_AF50C0(a1);
  v43 = *(_QWORD *)(a1 + 136);
  if ( v43 != a1 + 152 )
    _libc_free(v43, v42);
  if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 40), 24LL * *(unsigned int *)(a1 + 48), 8);
  return j_j___libc_free_0(a1, 184);
}
