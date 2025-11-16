// Function: sub_1E6B1C0
// Address: 0x1e6b1c0
//
__int64 __fastcall sub_1E6B1C0(size_t a1, unsigned __int8 *a2, size_t a3, __int64 a4, __int64 a5, int a6)
{
  size_t v6; // r8
  size_t v7; // rbx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r12
  unsigned int v10; // r15d
  unsigned __int64 v11; // rax
  size_t v12; // r13
  __m128i *v14; // rax
  __int64 i; // rdx
  __int64 v16; // rdx
  __int64 v17; // r9
  __int64 v18; // r13
  size_t v19; // r15
  __int64 v20; // r12
  int v21; // eax
  void *v22; // rdi
  unsigned int v23; // r14d
  size_t v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // r9
  size_t v28; // r8
  _QWORD *v29; // r11
  unsigned __int64 v30; // rax
  __int64 v31; // r12
  _QWORD *v32; // rax
  _QWORD *v33; // r12
  _BYTE *v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rax
  size_t v37; // r8
  unsigned int v38; // r10d
  _QWORD *v39; // r11
  __int64 v40; // rcx
  void *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // r12
  size_t v44; // r14
  _QWORD *v45; // rbx
  size_t v46; // rdx
  _BYTE *v47; // rax
  __int64 v48; // rax
  void *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r13
  __int64 v52; // rbx
  size_t v53; // r14
  unsigned __int64 v54; // rdi
  _QWORD *v55; // rdi
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 *v59; // r12
  __int64 *v60; // r14
  _BYTE *v61; // rdi
  size_t v62; // r13
  _BYTE *v63; // r15
  __int64 v64; // rax
  _QWORD *v65; // [rsp+0h] [rbp-A0h]
  size_t v66; // [rsp+0h] [rbp-A0h]
  size_t v67; // [rsp+0h] [rbp-A0h]
  unsigned int v68; // [rsp+8h] [rbp-98h]
  __int64 v69; // [rsp+8h] [rbp-98h]
  size_t v70; // [rsp+8h] [rbp-98h]
  __int64 v71; // [rsp+8h] [rbp-98h]
  size_t nb; // [rsp+10h] [rbp-90h]
  size_t n; // [rsp+10h] [rbp-90h]
  size_t nc; // [rsp+10h] [rbp-90h]
  size_t nd; // [rsp+10h] [rbp-90h]
  size_t na; // [rsp+10h] [rbp-90h]
  size_t ne; // [rsp+10h] [rbp-90h]
  size_t nf; // [rsp+10h] [rbp-90h]
  unsigned __int64 v79; // [rsp+18h] [rbp-88h]
  size_t v80; // [rsp+18h] [rbp-88h]
  unsigned int v81; // [rsp+18h] [rbp-88h]
  unsigned int v82; // [rsp+18h] [rbp-88h]
  unsigned int v83; // [rsp+18h] [rbp-88h]
  unsigned int v84; // [rsp+20h] [rbp-80h]
  size_t v85; // [rsp+20h] [rbp-80h]
  size_t v86; // [rsp+20h] [rbp-80h]
  size_t v87; // [rsp+20h] [rbp-80h]
  size_t v88; // [rsp+20h] [rbp-80h]
  size_t v89; // [rsp+20h] [rbp-80h]
  size_t v91; // [rsp+20h] [rbp-80h]
  size_t v93; // [rsp+20h] [rbp-80h]
  unsigned int v94; // [rsp+28h] [rbp-78h]
  size_t v95; // [rsp+28h] [rbp-78h]
  unsigned int v96; // [rsp+3Ch] [rbp-64h]
  size_t v97; // [rsp+48h] [rbp-58h] BYREF
  _QWORD *v98; // [rsp+50h] [rbp-50h] BYREF
  size_t v99; // [rsp+58h] [rbp-48h]
  _QWORD dest[8]; // [rsp+60h] [rbp-40h] BYREF

  v6 = a3;
  v7 = a1;
  v8 = *(unsigned int *)(a1 + 32);
  v9 = ((unsigned int)v8 & 0x7FFFFFFF) + 1;
  v96 = v8 | 0x80000000;
  v94 = v8 & 0x7FFFFFFF;
  v10 = v9;
  if ( (unsigned int)v8 >= (unsigned int)v9 )
  {
LABEL_2:
    v11 = *(unsigned int *)(a1 + 216);
    if ( (unsigned int)v11 >= (unsigned int)v9 )
      goto LABEL_3;
    goto LABEL_14;
  }
  if ( v9 >= v8 )
  {
    if ( v9 <= v8 )
      goto LABEL_2;
    if ( v9 > *(unsigned int *)(a1 + 36) )
    {
      sub_16CD150(a1 + 24, (const void *)(a1 + 40), v9, 16, a3, a6);
      v8 = *(unsigned int *)(a1 + 32);
      v6 = a3;
    }
    v14 = (__m128i *)(*(_QWORD *)(a1 + 24) + 16 * v8);
    for ( i = *(_QWORD *)(a1 + 24) + 16 * v9; (__m128i *)i != v14; ++v14 )
    {
      if ( v14 )
        *v14 = _mm_loadu_si128((const __m128i *)(a1 + 40));
    }
  }
  v11 = *(unsigned int *)(a1 + 216);
  *(_DWORD *)(a1 + 32) = v9;
  if ( (unsigned int)v11 >= (unsigned int)v9 )
    goto LABEL_3;
LABEL_14:
  if ( v9 < v11 )
  {
    v50 = *(_QWORD *)(a1 + 208);
    v51 = v50 + 40 * v9;
    if ( v50 + 40 * v11 != v51 )
    {
      v52 = v50 + 40 * v11;
      v53 = v6;
      do
      {
        v52 -= 40;
        v54 = *(_QWORD *)(v52 + 8);
        if ( v54 != v52 + 24 )
          _libc_free(v54);
      }
      while ( v51 != v52 );
      v7 = a1;
      v6 = v53;
    }
  }
  else
  {
    if ( v9 <= v11 )
      goto LABEL_3;
    if ( v9 > *(unsigned int *)(a1 + 220) )
    {
      v91 = v6;
      sub_1E6AA60((unsigned __int64 *)(a1 + 208), v9);
      v11 = *(unsigned int *)(a1 + 216);
      v6 = v91;
    }
    v16 = *(_QWORD *)(a1 + 208);
    v17 = v16 + 40 * v9;
    v18 = v16 + 40 * v11;
    if ( v17 != v18 )
    {
      v84 = v9;
      v19 = a1 + 232;
      v79 = v9;
      v20 = v16 + 40 * v9;
      do
      {
        if ( v18 )
        {
          v21 = *(_DWORD *)(v7 + 224);
          v22 = (void *)(v18 + 24);
          *(_DWORD *)(v18 + 16) = 0;
          *(_QWORD *)(v18 + 8) = v18 + 24;
          *(_DWORD *)v18 = v21;
          *(_DWORD *)(v18 + 20) = 4;
          v23 = *(_DWORD *)(v7 + 240);
          if ( v23 )
          {
            if ( v18 + 8 != v19 )
            {
              v24 = 4LL * v23;
              if ( v23 <= 4
                || (ne = v6,
                    sub_16CD150(v18 + 8, (const void *)(v18 + 24), v23, 4, v6, v17),
                    v22 = *(void **)(v18 + 8),
                    v6 = ne,
                    (v24 = 4LL * *(unsigned int *)(v7 + 240)) != 0) )
              {
                nb = v6;
                memcpy(v22, *(const void **)(v7 + 232), v24);
                v6 = nb;
              }
              *(_DWORD *)(v18 + 16) = v23;
            }
          }
        }
        v18 += 40;
      }
      while ( v20 != v18 );
      v10 = v84;
      v9 = v79;
    }
  }
  *(_DWORD *)(v7 + 216) = v10;
LABEL_3:
  v12 = v6;
  if ( !v6 )
    return v96;
  v85 = v6;
  v25 = sub_16D19C0(v7 + 120, a2, v6);
  v28 = v85;
  v29 = (_QWORD *)(*(_QWORD *)(v7 + 120) + 8LL * v25);
  if ( *v29 )
  {
    if ( *v29 != -8 )
    {
      v30 = *(unsigned int *)(v7 + 72);
      if ( v10 <= (unsigned int)v30 )
        goto LABEL_30;
      goto LABEL_44;
    }
    --*(_DWORD *)(v7 + 136);
  }
  v65 = v29;
  v68 = v25;
  n = v85;
  v80 = v85 + 17;
  v86 = v85 + 1;
  v36 = malloc(v28 + 17);
  v37 = n;
  v38 = v68;
  v39 = v65;
  v40 = v36;
  if ( v36 )
  {
LABEL_42:
    v41 = (void *)(v40 + 16);
    if ( v86 <= 1 )
      goto LABEL_43;
    goto LABEL_56;
  }
  if ( v80 || (v66 = n, nc = (size_t)v39, v48 = malloc(1u), v38 = v68, v39 = (_QWORD *)nc, v40 = 0, v37 = v66, !v48) )
  {
    v67 = v37;
    v71 = v40;
    nf = (size_t)v39;
    v83 = v38;
    sub_16BD1C0("Allocation failed", 1u);
    v38 = v83;
    v39 = (_QWORD *)nf;
    v40 = v71;
    v37 = v67;
    goto LABEL_42;
  }
  v41 = (void *)(v48 + 16);
  v40 = v48;
LABEL_56:
  v69 = v40;
  nd = (size_t)v39;
  v81 = v38;
  v89 = v37;
  v49 = memcpy(v41, a2, v37);
  v40 = v69;
  v39 = (_QWORD *)nd;
  v38 = v81;
  v37 = v89;
  v41 = v49;
LABEL_43:
  *((_BYTE *)v41 + v37) = 0;
  *(_QWORD *)v40 = v37;
  *(_BYTE *)(v40 + 8) = 0;
  *v39 = v40;
  ++*(_DWORD *)(v7 + 132);
  v87 = v37;
  sub_16D1CD0(v7 + 120, v38);
  v30 = *(unsigned int *)(v7 + 72);
  v28 = v87;
  if ( v10 <= (unsigned int)v30 )
    goto LABEL_30;
LABEL_44:
  if ( v9 < v30 )
  {
    v42 = *(_QWORD *)(v7 + 64) + 32 * v30;
    v43 = *(_QWORD *)(v7 + 64) + 32 * v9;
    if ( v42 != v43 )
    {
      v88 = v7;
      v44 = v28;
      v45 = (_QWORD *)v42;
      do
      {
        v45 -= 4;
        if ( (_QWORD *)*v45 != v45 + 2 )
          j_j___libc_free_0(*v45, v45[2] + 1LL);
      }
      while ( (_QWORD *)v43 != v45 );
      v7 = v88;
      v28 = v44;
    }
LABEL_51:
    *(_DWORD *)(v7 + 72) = v10;
    goto LABEL_30;
  }
  if ( v9 > v30 )
  {
    v57 = *(unsigned int *)(v7 + 76);
    if ( v9 > v57 )
    {
      v93 = v28;
      sub_12BE710(v7 + 64, v9, v57, v26, v28, v27);
      v30 = *(unsigned int *)(v7 + 72);
      v28 = v93;
    }
    v58 = 32 * v9 + *(_QWORD *)(v7 + 64);
    v59 = (__int64 *)(*(_QWORD *)(v7 + 64) + 32 * v30);
    if ( (__int64 *)v58 == v59 )
      goto LABEL_51;
    v82 = v10;
    v60 = (__int64 *)v58;
    na = v12;
    v70 = v28;
    while ( 1 )
    {
      if ( !v59 )
        goto LABEL_78;
      *v59 = (__int64)(v59 + 2);
      v63 = *(_BYTE **)(v7 + 80);
      v62 = *(_QWORD *)(v7 + 88);
      if ( &v63[v62] && !v63 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v98 = *(_QWORD **)(v7 + 88);
      if ( v62 > 0xF )
        break;
      v61 = (_BYTE *)*v59;
      if ( v62 != 1 )
      {
        if ( !v62 )
          goto LABEL_77;
        goto LABEL_84;
      }
      *v61 = *v63;
      v62 = (size_t)v98;
      v61 = (_BYTE *)*v59;
LABEL_77:
      v59[1] = v62;
      v61[v62] = 0;
LABEL_78:
      v59 += 4;
      if ( v60 == v59 )
      {
        v10 = v82;
        v12 = na;
        v28 = v70;
        goto LABEL_51;
      }
    }
    v64 = sub_22409D0(v59, &v98, 0);
    *v59 = v64;
    v61 = (_BYTE *)v64;
    v59[2] = (__int64)v98;
LABEL_84:
    memcpy(v61, v63, v62);
    v62 = (size_t)v98;
    v61 = (_BYTE *)*v59;
    goto LABEL_77;
  }
LABEL_30:
  v31 = 32LL * v94;
  if ( !a2 )
  {
    LOBYTE(dest[0]) = 0;
    v33 = (_QWORD *)(*(_QWORD *)(v7 + 64) + v31);
    v46 = 0;
    v98 = dest;
    v99 = 0;
LABEL_53:
    v47 = (_BYTE *)*v33;
    v33[1] = v46;
    v47[v46] = 0;
    v34 = v98;
    goto LABEL_38;
  }
  v97 = v28;
  v98 = dest;
  if ( v28 > 0xF )
  {
    v95 = v28;
    v56 = sub_22409D0(&v98, &v97, 0);
    v28 = v95;
    v98 = (_QWORD *)v56;
    v55 = (_QWORD *)v56;
    dest[0] = v97;
  }
  else
  {
    if ( v28 == 1 )
    {
      LOBYTE(dest[0]) = *a2;
      v32 = dest;
      goto LABEL_34;
    }
    v55 = dest;
  }
  memcpy(v55, a2, v28);
  v12 = v97;
  v32 = v98;
LABEL_34:
  v99 = v12;
  *((_BYTE *)v32 + v12) = 0;
  v33 = (_QWORD *)(*(_QWORD *)(v7 + 64) + v31);
  v34 = (_BYTE *)*v33;
  if ( v98 == dest )
  {
    v46 = v99;
    if ( v99 )
    {
      if ( v99 == 1 )
        *v34 = dest[0];
      else
        memcpy(v34, dest, v99);
      v46 = v99;
    }
    goto LABEL_53;
  }
  if ( v34 == (_BYTE *)(v33 + 2) )
  {
    *v33 = v98;
    v33[1] = v99;
    v33[2] = dest[0];
  }
  else
  {
    *v33 = v98;
    v35 = v33[2];
    v33[1] = v99;
    v33[2] = dest[0];
    if ( v34 )
    {
      v98 = v34;
      dest[0] = v35;
      goto LABEL_38;
    }
  }
  v98 = dest;
  v34 = dest;
LABEL_38:
  v99 = 0;
  *v34 = 0;
  if ( v98 != dest )
    j_j___libc_free_0(v98, dest[0] + 1LL);
  return v96;
}
