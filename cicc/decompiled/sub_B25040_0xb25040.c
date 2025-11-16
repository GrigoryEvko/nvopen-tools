// Function: sub_B25040
// Address: 0xb25040
//
void __fastcall sub_B25040(__m128i *a1, __int64 *a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 *v9; // r12
  __m128i *v10; // rbx
  bool v11; // al
  __int64 *v12; // rdx
  __m128i v13; // xmm0
  char v14; // al
  __m128i *v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rbx
  unsigned __int64 v18; // rcx
  char v19; // r11
  __int64 v20; // r8
  int v21; // esi
  __int64 *v22; // rbx
  unsigned int i; // edx
  _QWORD *v24; // r9
  __int64 v25; // r10
  unsigned int v26; // esi
  _DWORD *v27; // rbx
  __int64 v28; // r8
  unsigned __int64 v29; // rsi
  __int64 v30; // r9
  int v31; // edi
  unsigned int v32; // edx
  __int64 *v33; // rcx
  __int64 v34; // r10
  unsigned int v35; // edi
  _DWORD *v36; // rcx
  int v37; // edi
  __int64 v38; // rdi
  unsigned __int64 v39; // rcx
  __int64 *v40; // rax
  __int64 v41; // r8
  int v42; // esi
  int v43; // r11d
  __int64 *v44; // r10
  unsigned int v45; // edx
  _QWORD *v46; // r9
  __int64 v47; // rbx
  unsigned int v48; // edx
  unsigned int v49; // esi
  _DWORD *v50; // rbx
  __int64 v51; // r8
  unsigned __int64 v52; // rsi
  __int64 v53; // r9
  int v54; // edi
  __int64 *v55; // r11
  unsigned int j; // edx
  __int64 *v57; // rcx
  __int64 v58; // r10
  unsigned int v59; // edx
  unsigned int v60; // edi
  int v61; // eax
  __m128i v62; // xmm0
  unsigned int v63; // edx
  int v64; // ecx
  unsigned int v65; // esi
  unsigned __int64 v66; // rax
  unsigned int v67; // edx
  int v68; // ecx
  unsigned int v69; // r8d
  unsigned __int64 v70; // rdx
  unsigned int v71; // esi
  unsigned int v72; // eax
  int v73; // ecx
  unsigned int v74; // edx
  __int64 *v75; // r9
  unsigned __int64 v76; // rax
  unsigned int v77; // eax
  int v78; // edx
  unsigned int v79; // ecx
  __int64 *v80; // rcx
  unsigned __int64 v81; // rax
  unsigned int v82; // esi
  __m128i v83; // xmm4
  unsigned int v84; // edx
  unsigned int v85; // edx
  __int64 *v86; // rbx
  __int64 v87; // r9
  __int64 v88; // rcx
  __int64 v89; // r8
  bool v90; // al
  __m128i v91; // xmm0
  bool v92; // al
  __m128i v93; // xmm7
  __int128 v94; // [rsp-10h] [rbp-B0h]
  __int64 *m128i_i64; // [rsp+0h] [rbp-A0h]
  __int64 v96; // [rsp+8h] [rbp-98h]
  __m128i *v97; // [rsp+10h] [rbp-90h]
  int v98; // [rsp+1Ch] [rbp-84h]
  __int64 *v99; // [rsp+20h] [rbp-80h]
  int v100; // [rsp+30h] [rbp-70h]
  __int64 *v101; // [rsp+30h] [rbp-70h]
  int v102; // [rsp+30h] [rbp-70h]
  __m128i *v103; // [rsp+40h] [rbp-60h]
  __int64 *v105; // [rsp+58h] [rbp-48h] BYREF
  __int64 v106; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 v107; // [rsp+68h] [rbp-38h]

  v6 = (char *)a2 - (char *)a1;
  v97 = (__m128i *)a2;
  v96 = a3;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return;
  if ( !a3 )
  {
    v99 = a2;
    goto LABEL_140;
  }
  m128i_i64 = a1[1].m128i_i64;
  while ( 2 )
  {
    v106 = a4;
    --v96;
    v9 = v97[-1].m128i_i64;
    v10 = &a1[v6 >> 5];
    v107 = (unsigned __int64)a5;
    v11 = sub_B1DED0((__int64)&v106, m128i_i64, v10->m128i_i64);
    v12 = v97[-1].m128i_i64;
    if ( v11 )
    {
      if ( sub_B1DED0((__int64)&v106, v10->m128i_i64, v12) )
      {
        v13 = _mm_loadu_si128(a1);
        *a1 = _mm_loadu_si128(v10);
        *v10 = v13;
        goto LABEL_7;
      }
      v92 = sub_B1DED0((__int64)&v106, m128i_i64, v9);
      v91 = _mm_loadu_si128(a1);
      if ( !v92 )
      {
        v93 = _mm_loadu_si128(a1 + 1);
        a1[1] = v91;
        *a1 = v93;
        goto LABEL_7;
      }
LABEL_144:
      *a1 = _mm_loadu_si128(v97 - 1);
      v97[-1] = v91;
      goto LABEL_7;
    }
    if ( sub_B1DED0((__int64)&v106, m128i_i64, v12) )
    {
      v83 = _mm_loadu_si128(a1 + 1);
      a1[1] = _mm_loadu_si128(a1);
      *a1 = v83;
      goto LABEL_7;
    }
    v90 = sub_B1DED0((__int64)&v106, v10->m128i_i64, v9);
    v91 = _mm_loadu_si128(a1);
    if ( v90 )
      goto LABEL_144;
    *a1 = _mm_loadu_si128(v10);
    *v10 = v91;
LABEL_7:
    v14 = *(_BYTE *)(a4 + 8);
    v15 = v97;
    v103 = (__m128i *)m128i_i64;
    while ( 2 )
    {
      v16 = v103->m128i_i64[0];
      v99 = (__int64 *)v103;
      v17 = v103->m128i_i64[1];
      v106 = v103->m128i_i64[0];
      v18 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      v107 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      v19 = v14 & 1;
      if ( (v14 & 1) != 0 )
      {
        v20 = a4 + 16;
        v21 = 3;
      }
      else
      {
        v26 = *(_DWORD *)(a4 + 24);
        v20 = *(_QWORD *)(a4 + 16);
        if ( !v26 )
        {
          v72 = *(_DWORD *)(a4 + 8);
          ++*(_QWORD *)a4;
          v105 = 0;
          v73 = (v72 >> 1) + 1;
          goto LABEL_97;
        }
        v21 = v26 - 1;
      }
      v100 = 1;
      v22 = 0;
      for ( i = v21
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)
                  | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)))); ; i = v21 & v85 )
      {
        v24 = (_QWORD *)(v20 + 24LL * i);
        v25 = *v24;
        if ( v16 == *v24 && v18 == v24[1] )
        {
          v27 = v24 + 2;
          goto LABEL_22;
        }
        if ( v25 == -4096 )
          break;
        if ( v25 == -8192 && v24[1] == -8192 && !v22 )
          v22 = (__int64 *)(v20 + 24LL * i);
LABEL_138:
        v85 = v100 + i;
        ++v100;
      }
      if ( v24[1] != -4096 )
        goto LABEL_138;
      v72 = *(_DWORD *)(a4 + 8);
      v74 = 12;
      v26 = 4;
      if ( !v22 )
        v22 = v24;
      ++*(_QWORD *)a4;
      v105 = v22;
      v73 = (v72 >> 1) + 1;
      if ( !v19 )
      {
        v26 = *(_DWORD *)(a4 + 24);
LABEL_97:
        v74 = 3 * v26;
      }
      if ( 4 * v73 >= v74 )
      {
        v26 *= 2;
      }
      else if ( v26 - *(_DWORD *)(a4 + 12) - v73 > v26 >> 3 )
      {
        goto LABEL_100;
      }
      sub_B1DB20(a4, v26);
      sub_B1C410(a4, &v106, &v105);
      v16 = v106;
      v72 = *(_DWORD *)(a4 + 8);
LABEL_100:
      v75 = v105;
      *(_DWORD *)(a4 + 8) = (2 * (v72 >> 1) + 2) | v72 & 1;
      if ( *v75 != -4096 || v75[1] != -4096 )
        --*(_DWORD *)(a4 + 12);
      *v75 = v16;
      v76 = v107;
      v27 = v75 + 2;
      *((_DWORD *)v75 + 4) = 0;
      v75[1] = v76;
      v14 = *(_BYTE *)(a4 + 8);
      v19 = v14 & 1;
LABEL_22:
      v28 = a1->m128i_i64[0];
      v29 = a1->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v106 = a1->m128i_i64[0];
      v107 = v29;
      if ( v19 )
      {
        v30 = a4 + 16;
        v31 = 3;
      }
      else
      {
        v35 = *(_DWORD *)(a4 + 24);
        v30 = *(_QWORD *)(a4 + 16);
        if ( !v35 )
        {
          v77 = *(_DWORD *)(a4 + 8);
          ++*(_QWORD *)a4;
          v105 = 0;
          v78 = (v77 >> 1) + 1;
          goto LABEL_104;
        }
        v31 = v35 - 1;
      }
      v98 = 1;
      v101 = 0;
      v32 = v31
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4)
              | ((unsigned __int64)(((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4))));
      while ( 2 )
      {
        v33 = (__int64 *)(v30 + 24LL * v32);
        v34 = *v33;
        if ( v28 == *v33 && v29 == v33[1] )
        {
          v36 = v33 + 2;
          goto LABEL_36;
        }
        if ( v34 != -4096 )
        {
          if ( v34 == -8192 && v33[1] == -8192 )
          {
            if ( v101 )
              v33 = v101;
            v101 = v33;
          }
          goto LABEL_136;
        }
        if ( v33[1] != -4096 )
        {
LABEL_136:
          v84 = v98 + v32;
          ++v98;
          v32 = v31 & v84;
          continue;
        }
        break;
      }
      v35 = 4;
      if ( v101 )
        v33 = v101;
      v77 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v105 = v33;
      v79 = 12;
      v78 = (v77 >> 1) + 1;
      if ( !v19 )
      {
        v35 = *(_DWORD *)(a4 + 24);
LABEL_104:
        v79 = 3 * v35;
      }
      if ( 4 * v78 >= v79 )
      {
        v82 = 2 * v35;
      }
      else
      {
        if ( v35 - *(_DWORD *)(a4 + 12) - v78 > v35 >> 3 )
          goto LABEL_107;
        v82 = v35;
      }
      sub_B1DB20(a4, v82);
      sub_B1C410(a4, &v106, &v105);
      v28 = v106;
      v77 = *(_DWORD *)(a4 + 8);
LABEL_107:
      v80 = v105;
      *(_DWORD *)(a4 + 8) = (2 * (v77 >> 1) + 2) | v77 & 1;
      if ( *v80 != -4096 || v80[1] != -4096 )
        --*(_DWORD *)(a4 + 12);
      *v80 = v28;
      v81 = v107;
      v36 = v80 + 2;
      *v36 = 0;
      *((_QWORD *)v36 - 1) = v81;
      v14 = *(_BYTE *)(a4 + 8);
LABEL_36:
      v37 = *v36;
      if ( *a5 )
      {
        if ( *v27 < v37 )
          goto LABEL_38;
      }
      else if ( *v27 > v37 )
      {
LABEL_38:
        ++v103;
        continue;
      }
      break;
    }
    --v15;
    while ( 2 )
    {
      v38 = a1->m128i_i64[0];
      v39 = a1->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v106 = a1->m128i_i64[0];
      v107 = v39;
      LODWORD(v40) = v14 & 1;
      if ( (_DWORD)v40 )
      {
        v41 = a4 + 16;
        v42 = 3;
      }
      else
      {
        v49 = *(_DWORD *)(a4 + 24);
        v41 = *(_QWORD *)(a4 + 16);
        if ( !v49 )
        {
          v67 = *(_DWORD *)(a4 + 8);
          ++*(_QWORD *)a4;
          v105 = 0;
          v68 = (v67 >> 1) + 1;
          goto LABEL_82;
        }
        v42 = v49 - 1;
      }
      v43 = 1;
      v44 = 0;
      v45 = v42
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4)
              | ((unsigned __int64)(((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4))));
      while ( 2 )
      {
        v46 = (_QWORD *)(v41 + 24LL * v45);
        v47 = *v46;
        if ( v38 == *v46 && v39 == v46[1] )
        {
          v50 = v46 + 2;
          goto LABEL_55;
        }
        if ( v47 != -4096 )
        {
          if ( v47 == -8192 && v46[1] == -8192 && !v44 )
            v44 = (__int64 *)(v41 + 24LL * v45);
          goto LABEL_50;
        }
        if ( v46[1] != -4096 )
        {
LABEL_50:
          v48 = v43 + v45;
          ++v43;
          v45 = v42 & v48;
          continue;
        }
        break;
      }
      v67 = *(_DWORD *)(a4 + 8);
      v69 = 12;
      v49 = 4;
      if ( !v44 )
        v44 = v46;
      ++*(_QWORD *)a4;
      v105 = v44;
      v68 = (v67 >> 1) + 1;
      if ( !(_BYTE)v40 )
      {
        v49 = *(_DWORD *)(a4 + 24);
LABEL_82:
        v69 = 3 * v49;
      }
      if ( 4 * v68 >= v69 )
      {
        v49 *= 2;
      }
      else if ( v49 - *(_DWORD *)(a4 + 12) - v68 > v49 >> 3 )
      {
        goto LABEL_85;
      }
      sub_B1DB20(a4, v49);
      sub_B1C410(a4, &v106, &v105);
      v38 = v106;
      v67 = *(_DWORD *)(a4 + 8);
LABEL_85:
      *(_DWORD *)(a4 + 8) = (2 * (v67 >> 1) + 2) | v67 & 1;
      v40 = v105;
      if ( *v105 != -4096 || v105[1] != -4096 )
        --*(_DWORD *)(a4 + 12);
      *v40 = v38;
      v70 = v107;
      v50 = v40 + 2;
      *((_DWORD *)v40 + 4) = 0;
      v40[1] = v70;
      LOBYTE(v40) = *(_BYTE *)(a4 + 8) & 1;
LABEL_55:
      v51 = v15->m128i_i64[0];
      v52 = v15->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v106 = v15->m128i_i64[0];
      v107 = v52;
      if ( !(_BYTE)v40 )
      {
        v60 = *(_DWORD *)(a4 + 24);
        v53 = *(_QWORD *)(a4 + 16);
        if ( v60 )
        {
          v54 = v60 - 1;
          goto LABEL_57;
        }
        v63 = *(_DWORD *)(a4 + 8);
        ++*(_QWORD *)a4;
        v105 = 0;
        v64 = (v63 >> 1) + 1;
LABEL_75:
        v65 = 3 * v60;
LABEL_76:
        if ( v65 <= 4 * v64 )
        {
          v71 = 2 * v60;
        }
        else
        {
          if ( v60 - *(_DWORD *)(a4 + 12) - v64 > v60 >> 3 )
            goto LABEL_78;
          v71 = v60;
        }
        sub_B1DB20(a4, v71);
        sub_B1C410(a4, &v106, &v105);
        v51 = v106;
        v63 = *(_DWORD *)(a4 + 8);
LABEL_78:
        v57 = v105;
        *(_DWORD *)(a4 + 8) = (2 * (v63 >> 1) + 2) | v63 & 1;
        if ( *v57 != -4096 || v57[1] != -4096 )
          --*(_DWORD *)(a4 + 12);
        *v57 = v51;
        v66 = v107;
        *((_DWORD *)v57 + 4) = 0;
        v57[1] = v66;
        goto LABEL_68;
      }
      v53 = a4 + 16;
      v54 = 3;
LABEL_57:
      v102 = 1;
      v55 = 0;
      for ( j = v54
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4)
                  | ((unsigned __int64)(((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4)))); ; j = v54 & v59 )
      {
        v57 = (__int64 *)(v53 + 24LL * j);
        v58 = *v57;
        if ( v51 == *v57 && v52 == v57[1] )
          break;
        if ( v58 == -4096 )
        {
          if ( v57[1] == -4096 )
          {
            v63 = *(_DWORD *)(a4 + 8);
            v65 = 12;
            v60 = 4;
            if ( !v55 )
              v55 = v57;
            ++*(_QWORD *)a4;
            v105 = v55;
            v64 = (v63 >> 1) + 1;
            if ( !(_BYTE)v40 )
            {
              v60 = *(_DWORD *)(a4 + 24);
              goto LABEL_75;
            }
            goto LABEL_76;
          }
        }
        else if ( v58 == -8192 && v57[1] == -8192 && !v55 )
        {
          v55 = (__int64 *)(v53 + 24LL * j);
        }
        v59 = v102 + j;
        ++v102;
      }
LABEL_68:
      v61 = *((_DWORD *)v57 + 4);
      if ( *a5 )
      {
        if ( *v50 < v61 )
          goto LABEL_70;
      }
      else if ( *v50 > v61 )
      {
LABEL_70:
        v14 = *(_BYTE *)(a4 + 8);
        --v15;
        continue;
      }
      break;
    }
    if ( v103 < v15 )
    {
      v62 = _mm_loadu_si128(v103);
      *v103 = _mm_loadu_si128(v15);
      *v15 = v62;
      v14 = *(_BYTE *)(a4 + 8);
      goto LABEL_38;
    }
    v6 = (char *)v103 - (char *)a1;
    sub_B25040(v103, v97, v96, a4, a5);
    if ( (char *)v103 - (char *)a1 > 256 )
    {
      if ( v96 )
      {
        v97 = v103;
        continue;
      }
LABEL_140:
      v86 = v99;
      sub_B24F40(a1, v99, (unsigned __int64)v99, a4, (__int64)a5, a6);
      do
      {
        v86 -= 2;
        v88 = *v86;
        v89 = v86[1];
        *(__m128i *)v86 = _mm_loadu_si128(a1);
        *((_QWORD *)&v94 + 1) = a5;
        *(_QWORD *)&v94 = a4;
        sub_B24D50((__int64)a1, 0, ((char *)v86 - (char *)a1) >> 4, v88, v89, v87, v94);
      }
      while ( (char *)v86 - (char *)a1 > 16 );
    }
    break;
  }
}
