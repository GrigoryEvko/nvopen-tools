// Function: sub_3950CA0
// Address: 0x3950ca0
//
__int64 *__fastcall sub_3950CA0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 *v4; // r14
  __int64 v5; // rax
  unsigned int v6; // r13d
  __int64 v7; // rbx
  char *v8; // r12
  __int64 *v9; // r15
  char *v10; // r14
  __int64 v11; // r12
  __int64 v12; // rbx
  int v13; // r8d
  int v14; // r9d
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rcx
  _QWORD *v18; // rdx
  int v19; // r9d
  __int64 v20; // r15
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rcx
  _QWORD *v24; // rdx
  unsigned __int64 v26; // r11
  unsigned int v27; // ecx
  unsigned int v28; // edx
  unsigned __int64 v29; // rax
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // r11
  int v36; // esi
  unsigned int v37; // ecx
  int v38; // esi
  __int64 v39; // rdx
  __int64 v40; // rax
  size_t v41; // rdx
  unsigned __int64 v42; // r8
  unsigned int v43; // ebx
  unsigned int v44; // edx
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // r11
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rbx
  __int64 v49; // rax
  unsigned int v50; // r8d
  int v51; // edx
  unsigned int v52; // r10d
  int v53; // edx
  __int64 v54; // rdx
  unsigned __int64 v55; // r8
  __int64 v56; // rax
  __int64 v57; // [rsp+8h] [rbp-68h]
  unsigned __int64 v58; // [rsp+10h] [rbp-60h]
  unsigned int v59; // [rsp+10h] [rbp-60h]
  __int64 v60; // [rsp+18h] [rbp-58h]
  unsigned int v61; // [rsp+18h] [rbp-58h]
  int v62; // [rsp+18h] [rbp-58h]
  unsigned int v63; // [rsp+24h] [rbp-4Ch]
  size_t n; // [rsp+28h] [rbp-48h]
  size_t na; // [rsp+28h] [rbp-48h]
  size_t nb; // [rsp+28h] [rbp-48h]
  unsigned int nc; // [rsp+28h] [rbp-48h]
  size_t nd; // [rsp+28h] [rbp-48h]
  unsigned int v70; // [rsp+30h] [rbp-40h]
  unsigned __int64 v71; // [rsp+38h] [rbp-38h]

  v3 = a3;
  v4 = a1;
  v5 = sub_3950C20(a3, a2);
  v6 = *(_DWORD *)(v5 + 24);
  v7 = v5;
  if ( v6 )
  {
    v63 = (v6 + 63) >> 6;
    na = 8LL * v63;
    v71 = v63;
    v40 = malloc(na);
    v41 = na;
    v8 = (char *)v40;
    if ( !v40 )
    {
      if ( na || (v56 = malloc(1u), v41 = 0, !v56) )
      {
        nd = v41;
        sub_16BD1C0("Allocation failed", 1u);
        v41 = nd;
      }
      else
      {
        v8 = (char *)v56;
      }
    }
    memcpy(v8, *(const void **)(v7 + 8), v41);
  }
  else
  {
    v63 = 0;
    v8 = 0;
    v71 = 0;
  }
  *(_QWORD *)&v8[8 * (*(_DWORD *)v7 >> 6)] |= 1LL << *(_DWORD *)v7;
  if ( a1[1] != a2[1] )
  {
    n = v3;
    v9 = a1;
    v10 = v8;
    v11 = 8 * v71;
    while ( 1 )
    {
      v12 = sub_3950C20(n, v9);
      v15 = *(_DWORD *)(v12 + 24);
      if ( v6 > v15 )
        break;
LABEL_6:
      v16 = 0;
      if ( v71 )
      {
        do
        {
          v17 = *(_QWORD *)&v10[v16];
          v18 = (_QWORD *)(v16 + *(_QWORD *)(v12 + 8));
          v16 += 8;
          *v18 |= v17;
        }
        while ( v11 != v16 );
      }
      v9 = (__int64 *)v9[1];
      if ( v9[1] == a2[1] )
      {
        v8 = v10;
        v4 = v9;
        v3 = n;
        goto LABEL_10;
      }
    }
    v26 = *(_QWORD *)(v12 + 16);
    if ( v6 <= v26 << 6 )
      goto LABEL_15;
    v32 = 2 * v26;
    v57 = *(_QWORD *)(v12 + 16);
    if ( 2 * v26 < v71 )
      v32 = v71;
    v58 = v32;
    v60 = 8 * v32;
    v33 = (__int64)realloc(*(_QWORD *)(v12 + 8), 8 * v32, v32, 8 * (int)v32, v13, v14);
    v34 = v58;
    LODWORD(v35) = v57;
    if ( !v33 )
    {
      if ( v60 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v33 = 0;
        LODWORD(v35) = v57;
        v34 = v58;
      }
      else
      {
        v33 = sub_13A3880(1u);
        v34 = v58;
        LODWORD(v35) = v57;
      }
    }
    v36 = *(_DWORD *)(v12 + 24);
    *(_QWORD *)(v12 + 8) = v33;
    *(_QWORD *)(v12 + 16) = v34;
    v37 = (unsigned int)(v36 + 63) >> 6;
    if ( v34 > v37 )
    {
      v47 = v34 - v37;
      if ( v47 )
      {
        v59 = (unsigned int)(v36 + 63) >> 6;
        v62 = v35;
        memset((void *)(v33 + 8LL * v37), 0, 8 * v47);
        v36 = *(_DWORD *)(v12 + 24);
        v33 = *(_QWORD *)(v12 + 8);
        v37 = v59;
        LODWORD(v35) = v62;
      }
    }
    v38 = v36 & 0x3F;
    if ( v38 )
    {
      v35 = (unsigned int)v35;
      *(_QWORD *)(v33 + 8LL * (v37 - 1)) &= ~(-1LL << v38);
      v33 = *(_QWORD *)(v12 + 8);
      v39 = *(_QWORD *)(v12 + 16) - (unsigned int)v35;
      if ( !v39 )
        goto LABEL_29;
    }
    else
    {
      v35 = (unsigned int)v35;
      v39 = *(_QWORD *)(v12 + 16) - (unsigned int)v35;
      if ( !v39 )
      {
LABEL_29:
        v15 = *(_DWORD *)(v12 + 24);
        v28 = v15;
        if ( v6 <= v15 )
        {
LABEL_18:
          *(_DWORD *)(v12 + 24) = v6;
          if ( v6 < v28 )
          {
            v29 = *(_QWORD *)(v12 + 16);
            LOBYTE(v30) = v6;
            if ( v71 < v29 )
            {
              memset((void *)(v11 + *(_QWORD *)(v12 + 8)), 0, 8 * (v29 - v71));
              v30 = *(_DWORD *)(v12 + 24);
            }
            v31 = v30 & 0x3F;
            if ( v31 )
              *(_QWORD *)(*(_QWORD *)(v12 + 8) + 8LL * (v63 - 1)) &= ~(-1LL << v31);
          }
          goto LABEL_6;
        }
        v26 = *(_QWORD *)(v12 + 16);
LABEL_15:
        v27 = (v15 + 63) >> 6;
        if ( v26 > v27 )
        {
          v46 = v26 - v27;
          if ( v46 )
          {
            v61 = (v15 + 63) >> 6;
            memset((void *)(*(_QWORD *)(v12 + 8) + 8LL * v27), 0, 8 * v46);
            v15 = *(_DWORD *)(v12 + 24);
            v27 = v61;
          }
        }
        v28 = v15;
        if ( (v15 & 0x3F) != 0 )
        {
          *(_QWORD *)(*(_QWORD *)(v12 + 8) + 8LL * (v27 - 1)) &= ~(-1LL << (v15 & 0x3F));
          v28 = *(_DWORD *)(v12 + 24);
        }
        goto LABEL_18;
      }
    }
    memset((void *)(v33 + 8 * v35), 0, 8 * v39);
    goto LABEL_29;
  }
LABEL_10:
  v20 = sub_3950C20(v3, v4);
  v21 = *(_DWORD *)(v20 + 24);
  if ( v6 > v21 )
  {
    v42 = *(_QWORD *)(v20 + 16);
    if ( v6 > v42 << 6 )
    {
      v48 = 2 * v42;
      nb = *(_QWORD *)(v20 + 16);
      if ( 2 * v42 < v71 )
        v48 = v71;
      v49 = (__int64)realloc(*(_QWORD *)(v20 + 8), 8 * v48, 8 * (int)v48, (_DWORD)v42 << 6, v42, v19);
      v50 = nb;
      if ( !v49 )
      {
        if ( 8 * v48 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v49 = 0;
        }
        else
        {
          v49 = sub_13A3880(1u);
        }
        v50 = nb;
      }
      v51 = *(_DWORD *)(v20 + 24);
      *(_QWORD *)(v20 + 8) = v49;
      *(_QWORD *)(v20 + 16) = v48;
      v52 = (unsigned int)(v51 + 63) >> 6;
      if ( v52 < v48 )
      {
        nc = v50;
        v70 = (unsigned int)(v51 + 63) >> 6;
        memset((void *)(v49 + 8LL * v52), 0, 8 * (v48 - v52));
        v51 = *(_DWORD *)(v20 + 24);
        v49 = *(_QWORD *)(v20 + 8);
        v52 = v70;
        v50 = nc;
      }
      v53 = v51 & 0x3F;
      if ( v53 )
      {
        *(_QWORD *)(v49 + 8LL * (v52 - 1)) &= ~(-1LL << v53);
        v49 = *(_QWORD *)(v20 + 8);
      }
      v54 = *(_QWORD *)(v20 + 16) - v50;
      if ( v54 )
        memset((void *)(v49 + 8LL * v50), 0, 8 * v54);
      v21 = *(_DWORD *)(v20 + 24);
      v44 = v21;
      if ( v6 <= v21 )
        goto LABEL_39;
      v42 = *(_QWORD *)(v20 + 16);
    }
    v43 = (v21 + 63) >> 6;
    if ( v42 > v43 )
    {
      v55 = v42 - v43;
      if ( v55 )
      {
        memset((void *)(*(_QWORD *)(v20 + 8) + 8LL * v43), 0, 8 * v55);
        v21 = *(_DWORD *)(v20 + 24);
      }
    }
    v44 = v21;
    if ( (v21 & 0x3F) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(v20 + 8) + 8LL * (v43 - 1)) &= ~(-1LL << (v21 & 0x3F));
      v44 = *(_DWORD *)(v20 + 24);
    }
LABEL_39:
    *(_DWORD *)(v20 + 24) = v6;
    if ( v6 < v44 )
    {
      v45 = *(_QWORD *)(v20 + 16);
      if ( v71 < v45 )
      {
        memset((void *)(*(_QWORD *)(v20 + 8) + 8 * v71), 0, 8 * (v45 - v71));
        v6 = *(_DWORD *)(v20 + 24);
      }
      if ( (v6 & 0x3F) != 0 )
        *(_QWORD *)(*(_QWORD *)(v20 + 8) + 8LL * (v63 - 1)) &= ~(-1LL << (v6 & 0x3F));
    }
  }
  v22 = 0;
  if ( v71 )
  {
    do
    {
      v23 = *(_QWORD *)&v8[v22];
      v24 = (_QWORD *)(v22 + *(_QWORD *)(v20 + 8));
      v22 += 8;
      *v24 |= v23;
    }
    while ( v22 != 8 * v71 );
  }
  _libc_free((unsigned __int64)v8);
  return v4;
}
