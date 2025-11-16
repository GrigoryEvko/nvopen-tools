// Function: sub_29F6E50
// Address: 0x29f6e50
//
void __fastcall sub_29F6E50(__int64 a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // rax
  size_t v4; // r12
  const void *v5; // r13
  __int64 v6; // rbx
  size_t v7; // r14
  const void *v8; // r15
  size_t v9; // rdx
  int v10; // eax
  signed __int64 v11; // rax
  signed __int64 v12; // rax
  size_t v13; // r9
  const void *v14; // r10
  size_t v15; // rdx
  int v16; // eax
  signed __int64 v17; // rax
  signed __int64 v18; // rax
  __int64 v19; // rax
  __m128i *v20; // rbx
  __m128i *v21; // r15
  const void *v22; // r12
  size_t v23; // r14
  int v24; // eax
  unsigned __int64 v25; // rcx
  const void *v26; // r8
  signed __int64 v27; // rax
  signed __int64 v28; // rax
  __int64 v29; // rax
  size_t v30; // rdx
  const void *v31; // r9
  int v32; // eax
  signed __int64 v33; // rax
  signed __int64 v34; // rax
  size_t v35; // rbx
  size_t v36; // r13
  size_t v37; // r14
  size_t v38; // r14
  int v39; // eax
  signed __int64 v40; // rax
  __int64 v41; // rax
  size_t v42; // r9
  const void *v43; // r10
  size_t v44; // rdx
  int v45; // eax
  signed __int64 v46; // rax
  signed __int64 v47; // rax
  size_t v48; // r12
  int v49; // eax
  signed __int64 v50; // rax
  __int64 v51; // rdx
  signed __int64 v52; // rax
  signed __int64 v53; // rax
  __int64 v54; // r12
  const __m128i *v55; // r15
  __int64 v56; // r13
  __int64 v57; // rdi
  __int64 v58; // rsi
  __int64 v59; // rcx
  __m128i *v60; // rdx
  __m128i v61; // xmm2
  const void *v62; // r13
  __m128i *v63; // r15
  __int64 v64; // rax
  __int64 v65; // rax
  _BYTE *v66; // rax
  __int64 v67; // r12
  size_t v68; // rdx
  __int64 v69; // [rsp+8h] [rbp-C8h]
  __m128i *v70; // [rsp+10h] [rbp-C0h]
  __m128i *v71; // [rsp+18h] [rbp-B8h]
  size_t v72; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v73; // [rsp+20h] [rbp-B0h]
  size_t v74; // [rsp+20h] [rbp-B0h]
  size_t v75; // [rsp+28h] [rbp-A8h]
  size_t v76; // [rsp+28h] [rbp-A8h]
  size_t v77; // [rsp+28h] [rbp-A8h]
  __m128i *v78; // [rsp+28h] [rbp-A8h]
  size_t v79; // [rsp+28h] [rbp-A8h]
  size_t v80; // [rsp+28h] [rbp-A8h]
  size_t v81; // [rsp+28h] [rbp-A8h]
  size_t v82; // [rsp+28h] [rbp-A8h]
  size_t n; // [rsp+38h] [rbp-98h]
  const void *na; // [rsp+38h] [rbp-98h]
  size_t nb; // [rsp+38h] [rbp-98h]
  size_t nc; // [rsp+38h] [rbp-98h]
  size_t nd; // [rsp+38h] [rbp-98h]
  size_t ne; // [rsp+38h] [rbp-98h]
  size_t nf; // [rsp+38h] [rbp-98h]
  size_t ng; // [rsp+38h] [rbp-98h]
  const void *nh; // [rsp+38h] [rbp-98h]
  size_t ni; // [rsp+38h] [rbp-98h]
  size_t nj; // [rsp+38h] [rbp-98h]
  size_t nk; // [rsp+38h] [rbp-98h]
  __m128i *v96; // [rsp+40h] [rbp-90h]
  __int64 v97; // [rsp+48h] [rbp-88h]
  __m128i v98; // [rsp+50h] [rbp-80h] BYREF
  __int64 v99; // [rsp+60h] [rbp-70h]
  __m128i v100; // [rsp+70h] [rbp-60h] BYREF
  __m128i v101; // [rsp+80h] [rbp-50h] BYREF
  __int64 v102; // [rsp+90h] [rbp-40h]

  v3 = (__int64)a2->m128i_i64 - a1;
  v70 = a2;
  v69 = a3;
  if ( (__int64)a2->m128i_i64 - a1 <= 640 )
    return;
  if ( !a3 )
  {
    v71 = a2;
    goto LABEL_133;
  }
  while ( 2 )
  {
    --v69;
    v4 = *(_QWORD *)(a1 + 48);
    v5 = *(const void **)(a1 + 40);
    v6 = a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (v3 >> 3)) >> 1);
    v7 = *(_QWORD *)(v6 + 8);
    v8 = *(const void **)v6;
    v9 = v7;
    if ( v4 <= v7 )
      v9 = *(_QWORD *)(a1 + 48);
    if ( v9 )
    {
      n = v9;
      v10 = memcmp(*(const void **)(a1 + 40), *(const void **)v6, v9);
      v9 = n;
      if ( v10 )
      {
        if ( v10 >= 0 )
        {
LABEL_11:
          LODWORD(v12) = memcmp(v8, v5, v9);
          if ( (_DWORD)v12 )
            goto LABEL_14;
          goto LABEL_12;
        }
        goto LABEL_16;
      }
      v11 = v4 - v7;
      if ( (__int64)(v4 - v7) >= 0x80000000LL )
        goto LABEL_11;
    }
    else
    {
      v11 = v4 - v7;
      if ( (__int64)(v4 - v7) >= 0x80000000LL )
        goto LABEL_12;
    }
    if ( v11 > (__int64)0xFFFFFFFF7FFFFFFFLL && (int)v11 >= 0 )
    {
      if ( v9 )
        goto LABEL_11;
LABEL_12:
      v12 = v7 - v4;
      if ( (__int64)(v7 - v4) >= 0x80000000LL )
      {
LABEL_15:
        if ( *(_QWORD *)(a1 + 72) < *(_QWORD *)(v6 + 32) )
          goto LABEL_16;
LABEL_85:
        v42 = v70[-2].m128i_u64[0];
        v43 = (const void *)v70[-3].m128i_i64[1];
        v44 = v42;
        if ( v4 <= v42 )
          v44 = v4;
        if ( v44 )
        {
          v74 = v70[-2].m128i_u64[0];
          v80 = v44;
          nh = (const void *)v70[-3].m128i_i64[1];
          v45 = memcmp(v5, nh, v44);
          v43 = nh;
          v44 = v80;
          v42 = v74;
          if ( v45 )
          {
            if ( v45 < 0 )
              goto LABEL_110;
LABEL_93:
            v81 = v42;
            ni = (size_t)v43;
            LODWORD(v47) = memcmp(v43, v5, v44);
            v43 = (const void *)ni;
            v42 = v81;
            if ( (_DWORD)v47 )
              goto LABEL_96;
            goto LABEL_94;
          }
          v46 = v4 - v74;
          if ( (__int64)(v4 - v74) >= 0x80000000LL )
            goto LABEL_93;
        }
        else
        {
          v46 = v4 - v42;
          if ( (__int64)(v4 - v42) >= 0x80000000LL )
            goto LABEL_94;
        }
        if ( v46 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v46 < 0 )
          goto LABEL_110;
        if ( v44 )
          goto LABEL_93;
LABEL_94:
        v47 = v42 - v4;
        if ( (__int64)(v42 - v4) >= 0x80000000LL )
        {
LABEL_97:
          if ( *(_QWORD *)(a1 + 72) < v70[-1].m128i_i64[1] )
            goto LABEL_110;
LABEL_98:
          v48 = v42;
          if ( v7 <= v42 )
            v48 = v7;
          if ( v48 )
          {
            v82 = v42;
            nj = (size_t)v43;
            v49 = memcmp(v8, v43, v48);
            v43 = (const void *)nj;
            v42 = v82;
            if ( v49 )
            {
              if ( v49 < 0 )
                goto LABEL_73;
LABEL_103:
              nk = v42;
              LODWORD(v50) = memcmp(v43, v8, v48);
              v42 = nk;
              if ( (_DWORD)v50 )
                goto LABEL_106;
              goto LABEL_104;
            }
            v53 = v7 - v82;
            if ( (__int64)(v7 - v82) >= 0x80000000LL )
              goto LABEL_103;
          }
          else
          {
            v53 = v7 - v42;
            if ( (__int64)(v7 - v42) >= 0x80000000LL )
              goto LABEL_104;
          }
          if ( v53 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v53 < 0 )
            goto LABEL_73;
          if ( v48 )
            goto LABEL_103;
LABEL_104:
          v50 = v42 - v7;
          if ( (__int64)(v42 - v7) >= 0x80000000LL )
          {
LABEL_107:
            if ( *(_QWORD *)(v6 + 32) >= v70[-1].m128i_i64[1] )
              goto LABEL_29;
            goto LABEL_73;
          }
          if ( v50 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_29;
LABEL_106:
          if ( (int)v50 < 0 )
            goto LABEL_29;
          goto LABEL_107;
        }
        if ( v47 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_98;
LABEL_96:
        if ( (int)v47 < 0 )
          goto LABEL_98;
        goto LABEL_97;
      }
      if ( v12 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_85;
LABEL_14:
      if ( (int)v12 < 0 )
        goto LABEL_85;
      goto LABEL_15;
    }
LABEL_16:
    v13 = v70[-2].m128i_u64[0];
    v14 = (const void *)v70[-3].m128i_i64[1];
    v15 = v13;
    if ( v7 <= v13 )
      v15 = v7;
    if ( v15 )
    {
      v72 = v70[-2].m128i_u64[0];
      v75 = v15;
      na = (const void *)v70[-3].m128i_i64[1];
      v16 = memcmp(v8, na, v15);
      v14 = na;
      v15 = v75;
      v13 = v72;
      if ( v16 )
      {
        if ( v16 < 0 )
          goto LABEL_29;
LABEL_24:
        v76 = v13;
        nb = (size_t)v14;
        LODWORD(v18) = memcmp(v14, v8, v15);
        v14 = (const void *)nb;
        v13 = v76;
        if ( (_DWORD)v18 )
          goto LABEL_27;
        goto LABEL_25;
      }
      v17 = v7 - v72;
      if ( (__int64)(v7 - v72) >= 0x80000000LL )
        goto LABEL_24;
    }
    else
    {
      v17 = v7 - v13;
      if ( (__int64)(v7 - v13) >= 0x80000000LL )
        goto LABEL_25;
    }
    if ( v17 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v17 < 0 )
      goto LABEL_29;
    if ( v15 )
      goto LABEL_24;
LABEL_25:
    v18 = v13 - v7;
    if ( (__int64)(v13 - v7) >= 0x80000000LL )
      goto LABEL_28;
    if ( v18 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      goto LABEL_63;
LABEL_27:
    if ( (int)v18 < 0 )
      goto LABEL_63;
LABEL_28:
    if ( *(_QWORD *)(v6 + 32) >= v70[-1].m128i_i64[1] )
    {
LABEL_63:
      v38 = v13;
      if ( v4 <= v13 )
        v38 = v4;
      if ( v38 )
      {
        v79 = v13;
        nf = (size_t)v14;
        v39 = memcmp(v5, v14, v38);
        v14 = (const void *)nf;
        v13 = v79;
        if ( v39 )
        {
          if ( v39 >= 0 )
            goto LABEL_68;
LABEL_73:
          sub_22415E0((__m128i *)a1, (__m128i *)((char *)v70 - 40));
          v41 = *(_QWORD *)(a1 + 32);
          *(_QWORD *)(a1 + 32) = v70[-1].m128i_i64[1];
          v70[-1].m128i_i64[1] = v41;
          goto LABEL_30;
        }
        v52 = v4 - v79;
        if ( (__int64)(v4 - v79) >= 0x80000000LL )
          goto LABEL_68;
      }
      else
      {
        v52 = v4 - v13;
        if ( (__int64)(v4 - v13) >= 0x80000000LL )
        {
LABEL_69:
          v40 = v13 - v4;
          if ( (__int64)(v13 - v4) >= 0x80000000LL )
          {
LABEL_72:
            if ( *(_QWORD *)(a1 + 72) < v70[-1].m128i_i64[1] )
              goto LABEL_73;
LABEL_110:
            sub_22415E0((__m128i *)a1, (__m128i *)(a1 + 40));
            v51 = *(_QWORD *)(a1 + 72);
            *(_QWORD *)(a1 + 72) = *(_QWORD *)(a1 + 32);
            *(_QWORD *)(a1 + 32) = v51;
            goto LABEL_30;
          }
          if ( v40 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_110;
LABEL_71:
          if ( (int)v40 < 0 )
            goto LABEL_110;
          goto LABEL_72;
        }
      }
      if ( v52 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v52 < 0 )
        goto LABEL_73;
      if ( !v38 )
        goto LABEL_69;
LABEL_68:
      ng = v13;
      LODWORD(v40) = memcmp(v14, v5, v38);
      v13 = ng;
      if ( (_DWORD)v40 )
        goto LABEL_71;
      goto LABEL_69;
    }
LABEL_29:
    sub_22415E0((__m128i *)a1, (__m128i *)v6);
    v19 = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 32) = *(_QWORD *)(v6 + 32);
    *(_QWORD *)(v6 + 32) = v19;
LABEL_30:
    v20 = (__m128i *)(a1 + 40);
    v21 = v70;
    v22 = *(const void **)a1;
    v23 = *(_QWORD *)(a1 + 8);
    while ( 1 )
    {
      v25 = v20->m128i_u64[1];
      v30 = v23;
      v31 = (const void *)v20->m128i_i64[0];
      v71 = v20;
      if ( v25 <= v23 )
        v30 = v20->m128i_u64[1];
      if ( !v30 )
      {
        v33 = v25 - v23;
        if ( (__int64)(v25 - v23) >= 0x80000000LL )
          goto LABEL_52;
        goto LABEL_48;
      }
      v73 = v20->m128i_u64[1];
      v77 = v30;
      nd = v20->m128i_i64[0];
      v32 = memcmp(v31, v22, v30);
      v31 = (const void *)nd;
      v30 = v77;
      v25 = v73;
      if ( !v32 )
        break;
      if ( v32 >= 0 )
        goto LABEL_51;
LABEL_42:
      v20 = (__m128i *)((char *)v20 + 40);
    }
    v33 = v73 - v23;
    if ( (__int64)(v73 - v23) >= 0x80000000LL )
      goto LABEL_51;
LABEL_48:
    if ( v33 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v33 < 0 )
      goto LABEL_42;
    if ( !v30 )
      goto LABEL_52;
LABEL_51:
    ne = v25;
    LODWORD(v34) = memcmp(v22, v31, v30);
    v25 = ne;
    if ( (_DWORD)v34 )
      goto LABEL_54;
LABEL_52:
    v34 = v23 - v25;
    if ( (__int64)(v23 - v25) >= 0x80000000LL )
      goto LABEL_55;
    v25 = 0xFFFFFFFF7FFFFFFFLL;
    if ( v34 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      goto LABEL_56;
LABEL_54:
    if ( (int)v34 < 0 )
      goto LABEL_56;
LABEL_55:
    if ( v20[2].m128i_i64[0] < *(_QWORD *)(a1 + 32) )
      goto LABEL_42;
LABEL_56:
    v78 = v20;
    v21 = (__m128i *)((char *)v21 - 40);
    v35 = v23;
    while ( 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v36 = v21->m128i_u64[1];
          v37 = v35;
          v26 = (const void *)v21->m128i_i64[0];
          if ( v36 <= v35 )
            v37 = v21->m128i_u64[1];
          if ( !v37 )
            break;
          nc = v21->m128i_i64[0];
          v24 = memcmp(v22, (const void *)v21->m128i_i64[0], v37);
          v26 = (const void *)nc;
          if ( !v24 )
          {
            v27 = v35 - v36;
            if ( (__int64)(v35 - v36) >= 0x80000000LL )
              goto LABEL_36;
            v25 = 0xFFFFFFFF7FFFFFFFLL;
            if ( v27 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_62;
            goto LABEL_34;
          }
          if ( v24 >= 0 )
            goto LABEL_36;
          v21 = (__m128i *)((char *)v21 - 40);
        }
        v27 = v35 - v36;
        if ( (__int64)(v35 - v36) >= 0x80000000LL )
          goto LABEL_37;
        v25 = 0xFFFFFFFF7FFFFFFFLL;
        if ( v27 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          break;
LABEL_62:
        v21 = (__m128i *)((char *)v21 - 40);
      }
LABEL_34:
      if ( (int)v27 < 0 )
        goto LABEL_62;
      if ( v37 )
      {
LABEL_36:
        LODWORD(v28) = memcmp(v26, v22, v37);
        if ( (_DWORD)v28 )
          goto LABEL_39;
      }
LABEL_37:
      v28 = v36 - v35;
      if ( (__int64)(v36 - v35) >= 0x80000000LL )
        goto LABEL_74;
      v25 = 0xFFFFFFFF7FFFFFFFLL;
      if ( v28 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        break;
LABEL_39:
      if ( (int)v28 < 0 )
        break;
LABEL_74:
      if ( *(_QWORD *)(a1 + 32) >= v21[2].m128i_i64[0] )
        break;
      v21 = (__m128i *)((char *)v21 - 40);
    }
    v20 = v78;
    if ( v78 < v21 )
    {
      sub_22415E0(v78, v21);
      v29 = v78[2].m128i_i64[0];
      v78[2].m128i_i64[0] = v21[2].m128i_i64[0];
      v21[2].m128i_i64[0] = v29;
      v22 = *(const void **)a1;
      v23 = *(_QWORD *)(a1 + 8);
      goto LABEL_42;
    }
    sub_29F6E50(v78, v70, v69, v25, v26);
    v3 = (__int64)v78->m128i_i64 - a1;
    if ( (__int64)v78->m128i_i64 - a1 <= 640 )
      return;
    if ( v69 )
    {
      v70 = v78;
      continue;
    }
    break;
  }
LABEL_133:
  v54 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v3 >> 3) - 2) >> 1;
  v55 = (const __m128i *)(a1 + 40 * v54 + 16);
  v56 = 0xCCCCCCCCCCCCCCCDLL * (v3 >> 3);
  while ( 2 )
  {
    v60 = (__m128i *)v55[-1].m128i_i64[0];
    if ( v60 == v55 )
    {
      v59 = v55[1].m128i_i64[0];
      v61 = _mm_loadu_si128(v55);
      v100.m128i_i64[0] = (__int64)&v101;
      v58 = v55[-1].m128i_i64[1];
      v55->m128i_i8[0] = 0;
      v55[-1].m128i_i64[1] = 0;
      v99 = v59;
      v98 = v61;
LABEL_142:
      v101 = _mm_load_si128(&v98);
    }
    else
    {
      v57 = v55->m128i_i64[0];
      v58 = v55[-1].m128i_i64[1];
      v55[-1].m128i_i64[0] = (__int64)v55;
      v59 = v55[1].m128i_i64[0];
      v98.m128i_i64[0] = v57;
      v55[-1].m128i_i64[1] = 0;
      v55->m128i_i8[0] = 0;
      v99 = v59;
      v100.m128i_i64[0] = (__int64)&v101;
      if ( v60 == &v98 )
        goto LABEL_142;
      v100.m128i_i64[0] = (__int64)v60;
      v101.m128i_i64[0] = v57;
    }
    v102 = v59;
    v100.m128i_i64[1] = v58;
    v98.m128i_i8[0] = 0;
    sub_29F5D00((char *)a1, v54, v56, &v100);
    if ( (__m128i *)v100.m128i_i64[0] != &v101 )
      j_j___libc_free_0(v100.m128i_u64[0]);
    if ( v54 )
    {
      --v54;
      v55 = (const __m128i *)((char *)v55 - 40);
      continue;
    }
    break;
  }
  v62 = (const void *)(a1 + 16);
  v63 = (__m128i *)((char *)v71 - 24);
  do
  {
    v96 = &v98;
    if ( (__m128i *)v63[-1].m128i_i64[0] == v63 )
    {
      v98 = _mm_loadu_si128(v63);
    }
    else
    {
      v96 = (__m128i *)v63[-1].m128i_i64[0];
      v98.m128i_i64[0] = v63->m128i_i64[0];
    }
    v64 = v63[-1].m128i_i64[1];
    v63[-1].m128i_i64[0] = (__int64)v63;
    v63[-1].m128i_i64[1] = 0;
    v97 = v64;
    v65 = v63[1].m128i_i64[0];
    v63->m128i_i8[0] = 0;
    v99 = v65;
    if ( *(const void **)a1 == v62 )
    {
      v68 = *(_QWORD *)(a1 + 8);
      if ( v68 )
      {
        if ( v68 == 1 )
          v63->m128i_i8[0] = *(_BYTE *)(a1 + 16);
        else
          memcpy(v63, v62, v68);
        v68 = *(_QWORD *)(a1 + 8);
      }
      v63[-1].m128i_i64[1] = v68;
      v63->m128i_i8[v68] = 0;
      v66 = *(_BYTE **)a1;
    }
    else
    {
      v63[-1].m128i_i64[0] = *(_QWORD *)a1;
      v63[-1].m128i_i64[1] = *(_QWORD *)(a1 + 8);
      v63->m128i_i64[0] = *(_QWORD *)(a1 + 16);
      v66 = (_BYTE *)(a1 + 16);
      *(_QWORD *)a1 = v62;
    }
    *(_QWORD *)(a1 + 8) = 0;
    *v66 = 0;
    v63[1].m128i_i64[0] = *(_QWORD *)(a1 + 32);
    v100.m128i_i64[0] = (__int64)&v101;
    if ( v96 == &v98 )
    {
      v101 = _mm_load_si128(&v98);
    }
    else
    {
      v100.m128i_i64[0] = (__int64)v96;
      v101.m128i_i64[0] = v98.m128i_i64[0];
    }
    v67 = (__int64)v63[-1].m128i_i64 - a1;
    v98.m128i_i8[0] = 0;
    v100.m128i_i64[1] = v97;
    v102 = v99;
    sub_29F5D00((char *)a1, 0, 0xCCCCCCCCCCCCCCCDLL * (v67 >> 3), &v100);
    if ( (__m128i *)v100.m128i_i64[0] != &v101 )
      j_j___libc_free_0(v100.m128i_u64[0]);
    v63 = (__m128i *)((char *)v63 - 40);
  }
  while ( v67 > 40 );
}
