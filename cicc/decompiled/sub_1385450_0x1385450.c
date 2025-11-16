// Function: sub_1385450
// Address: 0x1385450
//
void __fastcall sub_1385450(__int64 a1, int a2, __int64 a3, __int64 a4, unsigned __int8 a5, __int64 a6, __int64 a7)
{
  unsigned __int8 v11; // cl
  unsigned int v12; // esi
  __int64 v13; // r10
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  int v16; // eax
  size_t v17; // rdx
  unsigned int i; // eax
  size_t v19; // r12
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rdi
  __int64 v23; // rsi
  unsigned int v24; // esi
  __int64 v25; // r11
  __int64 v26; // r10
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  __int64 *v29; // rax
  unsigned int j; // r8d
  __int64 *v31; // rdx
  __int64 v32; // rdi
  unsigned int v33; // r8d
  int v34; // eax
  int v35; // edi
  __m128i v36; // xmm0
  __int64 v37; // rax
  int v38; // esi
  size_t v39; // rdx
  int v40; // edi
  int v41; // edi
  __int64 v42; // r10
  int v43; // r11d
  __int64 *v44; // r8
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // rsi
  unsigned int m; // esi
  __int64 v49; // r9
  unsigned int v50; // esi
  int v51; // edi
  int v52; // edi
  __int64 v53; // rsi
  __int64 v54; // rdx
  const void *v55; // r8
  signed __int64 v56; // rdx
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rsi
  bool v59; // cf
  unsigned __int64 v60; // rax
  __int64 v61; // r12
  __int64 v62; // rax
  char *v63; // r9
  __int64 v64; // r12
  char *v65; // rax
  __int64 v66; // rbx
  int v67; // esi
  int v68; // esi
  __int64 v69; // rdi
  __int64 *v70; // r8
  int v71; // r10d
  unsigned int k; // edx
  __int64 v73; // r9
  unsigned int v74; // edx
  char *v75; // rax
  size_t v76; // rsi
  int v77; // [rsp+Ch] [rbp-64h]
  unsigned __int8 v78; // [rsp+Ch] [rbp-64h]
  unsigned __int8 v79; // [rsp+Ch] [rbp-64h]
  unsigned __int8 v80; // [rsp+Ch] [rbp-64h]
  unsigned __int8 v81; // [rsp+Ch] [rbp-64h]
  int n; // [rsp+10h] [rbp-60h]
  size_t na; // [rsp+10h] [rbp-60h]
  size_t nb; // [rsp+10h] [rbp-60h]
  size_t nc; // [rsp+10h] [rbp-60h]
  size_t nd; // [rsp+10h] [rbp-60h]
  size_t ne; // [rsp+10h] [rbp-60h]
  void *srca; // [rsp+18h] [rbp-58h]
  char *srcb; // [rsp+18h] [rbp-58h]
  size_t v91; // [rsp+28h] [rbp-48h] BYREF
  __m128i v92; // [rsp+30h] [rbp-40h] BYREF

  v11 = a5;
  if ( a1 == a3 && (_DWORD)a4 == a2 )
    return;
  v12 = *(_DWORD *)(a6 + 24);
  v92.m128i_i64[0] = a3;
  v92.m128i_i64[1] = a4;
  if ( !v12 )
  {
    ++*(_QWORD *)a6;
    goto LABEL_42;
  }
  v13 = *(_QWORD *)(a6 + 8);
  n = 1;
  v14 = ((((unsigned int)(37 * a4) | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * a4) << 32)) >> 22)
      ^ (((unsigned int)(37 * a4) | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * a4) << 32));
  v15 = ((9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13)))) >> 15)
      ^ (9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13))));
  v16 = ((v15 - 1 - (v15 << 27)) >> 31) ^ (v15 - 1 - ((_DWORD)v15 << 27));
  v17 = 0;
  for ( i = (v12 - 1) & v16; ; i = (v12 - 1) & v21 )
  {
    v19 = v13 + 48LL * i;
    v20 = *(_QWORD *)v19;
    if ( *(_QWORD *)v19 == a3 && *(_DWORD *)(v19 + 8) == (_DWORD)a4 )
      break;
    if ( v20 == -8 )
    {
      if ( *(_DWORD *)(v19 + 8) == -1 )
      {
        v34 = *(_DWORD *)(a6 + 16);
        if ( !v17 )
          v17 = v19;
        ++*(_QWORD *)a6;
        v35 = v34 + 1;
        if ( 4 * (v34 + 1) < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a6 + 20) - v35 > v12 >> 3 )
          {
LABEL_25:
            *(_DWORD *)(a6 + 16) = v35;
            if ( *(_QWORD *)v17 != -8 || *(_DWORD *)(v17 + 8) != -1 )
              --*(_DWORD *)(a6 + 20);
            v36 = _mm_loadu_si128(&v92);
            *(_QWORD *)(v17 + 16) = 0;
            v26 = v17 + 16;
            v37 = 1;
            *(_QWORD *)(v17 + 24) = 0;
            *(_QWORD *)(v17 + 32) = 0;
            *(_DWORD *)(v17 + 40) = 0;
            *(__m128i *)v17 = v36;
            goto LABEL_28;
          }
          v79 = a5;
LABEL_43:
          nc = a6;
          sub_1384E70(a6, v12);
          sub_1383DA0(nc, v92.m128i_i64, &v91);
          a6 = nc;
          v17 = v91;
          v11 = v79;
          v35 = *(_DWORD *)(nc + 16) + 1;
          goto LABEL_25;
        }
LABEL_42:
        v79 = a5;
        v12 *= 2;
        goto LABEL_43;
      }
    }
    else if ( v20 == -16 && *(_DWORD *)(v19 + 8) == -2 && !v17 )
    {
      v17 = v13 + 48LL * i;
    }
    v21 = n + i;
    ++n;
  }
  v24 = *(_DWORD *)(v19 + 40);
  v25 = *(_QWORD *)(v19 + 24);
  v26 = v19 + 16;
  if ( !v24 )
  {
    v17 = v19;
    v37 = *(_QWORD *)(v19 + 16) + 1LL;
LABEL_28:
    *(_QWORD *)(v17 + 16) = v37;
    v38 = 0;
    goto LABEL_29;
  }
  v77 = 1;
  v27 = ((((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * a2) << 32)) >> 22)
      ^ (((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * a2) << 32));
  v28 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
      ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
  na = ((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - (v28 << 27));
  v29 = 0;
  for ( j = (v24 - 1) & (((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - ((_DWORD)v28 << 27))); ; j = (v24 - 1) & v33 )
  {
    v31 = (__int64 *)(v25 + 24LL * j);
    v32 = *v31;
    if ( *v31 == a1 && *((_DWORD *)v31 + 2) == a2 )
      break;
    if ( v32 == -8 )
    {
      if ( *((_DWORD *)v31 + 2) == -1 )
      {
        v51 = *(_DWORD *)(v19 + 32);
        if ( !v29 )
          v29 = (__int64 *)(v25 + 24LL * j);
        ++*(_QWORD *)(v19 + 16);
        v52 = v51 + 1;
        if ( 4 * v52 < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(v19 + 36) - v52 > v24 >> 3 )
            goto LABEL_50;
          v81 = v11;
          sub_1385190(v19 + 16, v24);
          v67 = *(_DWORD *)(v19 + 40);
          if ( v67 )
          {
            v68 = v67 - 1;
            v70 = 0;
            v11 = v81;
            v71 = 1;
            for ( k = v68 & na; ; k = v68 & v74 )
            {
              v69 = *(_QWORD *)(v19 + 24);
              v29 = (__int64 *)(v69 + 24LL * k);
              v73 = *v29;
              if ( *v29 == a1 && *((_DWORD *)v29 + 2) == a2 )
              {
                v52 = *(_DWORD *)(v19 + 32) + 1;
                goto LABEL_50;
              }
              if ( v73 == -8 )
              {
                if ( *((_DWORD *)v29 + 2) == -1 )
                {
                  v52 = *(_DWORD *)(v19 + 32) + 1;
                  if ( v70 )
                    v29 = v70;
                  goto LABEL_50;
                }
              }
              else if ( v73 == -16 && *((_DWORD *)v29 + 2) == -2 && !v70 )
              {
                v70 = (__int64 *)(v69 + 24LL * k);
              }
              v74 = v71 + k;
              ++v71;
            }
          }
          v39 = v19;
          goto LABEL_115;
        }
        v38 = 2 * v24;
        v17 = v19;
LABEL_29:
        nb = v17;
        v78 = v11;
        sub_1385190(v26, v38);
        v39 = nb;
        v40 = *(_DWORD *)(nb + 40);
        if ( v40 )
        {
          v41 = v40 - 1;
          v42 = *(_QWORD *)(nb + 24);
          v11 = v78;
          v43 = 1;
          v44 = 0;
          v45 = ((((unsigned int)(37 * a2)
                 | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * a2) << 32)) >> 22)
              ^ (((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * a2) << 32));
          v46 = 9 * (((v45 - 1 - (v45 << 13)) >> 8) ^ (v45 - 1 - (v45 << 13)));
          v47 = ((v46 >> 15) ^ v46) - 1 - (((v46 >> 15) ^ v46) << 27);
          for ( m = v41 & ((v47 >> 31) ^ v47); ; m = v41 & v50 )
          {
            v29 = (__int64 *)(v42 + 24LL * m);
            v49 = *v29;
            if ( *v29 == a1 && *((_DWORD *)v29 + 2) == a2 )
            {
              v19 = nb;
              v52 = *(_DWORD *)(nb + 32) + 1;
              goto LABEL_50;
            }
            if ( v49 == -8 )
            {
              if ( *((_DWORD *)v29 + 2) == -1 )
              {
                v19 = nb;
                v52 = *(_DWORD *)(nb + 32) + 1;
                if ( v44 )
                  v29 = v44;
LABEL_50:
                *(_DWORD *)(v19 + 32) = v52;
                if ( *v29 != -8 || *((_DWORD *)v29 + 2) != -1 )
                  --*(_DWORD *)(v19 + 36);
                *v29 = a1;
                v22 = v11;
                *((_DWORD *)v29 + 2) = a2;
                v29[2] = 0;
                if ( v11 > 6u )
LABEL_53:
                  sub_1381520(v22, (char)"bitset::test");
                v53 = 1LL << v11;
LABEL_55:
                v29[2] = v53;
                v54 = *(_QWORD *)(a7 + 8);
                if ( v54 != *(_QWORD *)(a7 + 16) )
                {
                  if ( v54 )
                  {
                    *(_QWORD *)v54 = a1;
                    *(_DWORD *)(v54 + 8) = a2;
                    *(_QWORD *)(v54 + 16) = a3;
                    *(_DWORD *)(v54 + 24) = a4;
                    *(_BYTE *)(v54 + 32) = v11;
                    v54 = *(_QWORD *)(a7 + 8);
                  }
                  *(_QWORD *)(a7 + 8) = v54 + 40;
                  return;
                }
                v55 = *(const void **)a7;
                v56 = v54 - *(_QWORD *)a7;
                v57 = 0xCCCCCCCCCCCCCCCDLL * (v56 >> 3);
                if ( v57 == 0x333333333333333LL )
                  sub_4262D8((__int64)"vector::_M_realloc_insert");
                v58 = 1;
                if ( v57 )
                  v58 = 0xCCCCCCCCCCCCCCCDLL * (v56 >> 3);
                v59 = __CFADD__(v58, v57);
                v60 = v58 - 0x3333333333333333LL * (v56 >> 3);
                if ( v59 )
                {
                  v61 = 0x7FFFFFFFFFFFFFF8LL;
                  goto LABEL_69;
                }
                if ( v60 )
                {
                  if ( v60 > 0x333333333333333LL )
                    v60 = 0x333333333333333LL;
                  v61 = 40 * v60;
LABEL_69:
                  v80 = v11;
                  nd = v56;
                  srca = *(void **)a7;
                  v62 = sub_22077B0(v61);
                  v55 = srca;
                  v56 = nd;
                  v11 = v80;
                  v63 = (char *)v62;
                  v64 = v62 + v61;
                }
                else
                {
                  v64 = 0;
                  v63 = 0;
                }
                v65 = &v63[v56];
                if ( &v63[v56] )
                {
                  *(_QWORD *)v65 = a1;
                  *((_DWORD *)v65 + 2) = a2;
                  *((_QWORD *)v65 + 2) = a3;
                  *((_DWORD *)v65 + 6) = a4;
                  v65[32] = v11;
                }
                v66 = (__int64)&v63[v56 + 40];
                if ( v56 > 0 )
                {
                  ne = (size_t)v55;
                  v75 = (char *)memmove(v63, v55, v56);
                  v55 = (const void *)ne;
                  v63 = v75;
                  v76 = *(_QWORD *)(a7 + 16) - ne;
                }
                else
                {
                  if ( !v55 )
                  {
LABEL_74:
                    *(_QWORD *)a7 = v63;
                    *(_QWORD *)(a7 + 8) = v66;
                    *(_QWORD *)(a7 + 16) = v64;
                    return;
                  }
                  v76 = *(_QWORD *)(a7 + 16) - (_QWORD)v55;
                }
                srcb = v63;
                j_j___libc_free_0(v55, v76);
                v63 = srcb;
                goto LABEL_74;
              }
            }
            else if ( v49 == -16 && *((_DWORD *)v29 + 2) == -2 && !v44 )
            {
              v44 = (__int64 *)(v42 + 24LL * m);
            }
            v50 = v43 + m;
            ++v43;
          }
        }
LABEL_115:
        ++*(_DWORD *)(v39 + 32);
        BUG();
      }
    }
    else if ( v32 == -16 && *((_DWORD *)v31 + 2) == -2 && !v29 )
    {
      v29 = (__int64 *)(v25 + 24LL * j);
    }
    v33 = v77 + j;
    ++v77;
  }
  v22 = v11;
  if ( v11 > 6u )
    goto LABEL_53;
  v23 = v31[2];
  if ( ((1LL << v11) & v23) == 0 )
  {
    v53 = (1LL << v11) | v23;
    v29 = (__int64 *)(v25 + 24LL * j);
    goto LABEL_55;
  }
}
