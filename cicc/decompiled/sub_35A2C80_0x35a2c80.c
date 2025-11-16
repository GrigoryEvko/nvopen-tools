// Function: sub_35A2C80
// Address: 0x35a2c80
//
__int64 __fastcall sub_35A2C80(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 i; // r15
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdi
  __int32 v17; // ebx
  __int64 v18; // r12
  __int64 *v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rax
  char v25; // al
  __int64 v26; // r8
  bool v27; // zf
  __int64 *v28; // rax
  int v29; // ecx
  unsigned int v30; // esi
  int v31; // edx
  __int64 v32; // rdx
  __int64 v33; // r12
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rbx
  __int64 *v40; // rax
  int v41; // ecx
  unsigned int v42; // esi
  int v43; // edx
  __int64 v44; // rdx
  int v45; // ecx
  int v46; // r8d
  int v47; // esi
  unsigned int j; // eax
  _QWORD *v49; // rcx
  unsigned int v50; // eax
  int v51; // eax
  int v52; // eax
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 *v55; // r13
  int v56; // r14d
  __int64 *v57; // rbx
  unsigned __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rsi
  __int64 v61; // rax
  unsigned int v62; // ecx
  __int64 *v63; // rdx
  __int64 v64; // r9
  __int64 v65; // rax
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  __int64 *v70; // r12
  __int64 v71; // rbx
  __int64 v72; // rdi
  __int64 v73; // rdi
  __int64 v74; // r12
  __int64 v75; // rbx
  __int64 v76; // r14
  int v77; // esi
  unsigned int v78; // edx
  int *v79; // rcx
  int v80; // edi
  int v82; // ecx
  unsigned __int64 v83; // rax
  _QWORD *v84; // rdi
  __int64 v85; // r12
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 v88; // r8
  __int64 v89; // r9
  int v90; // r12d
  __int64 *v91; // rax
  __int64 *v92; // rax
  int v93; // edx
  int v94; // r10d
  int v95; // ecx
  unsigned int v96; // esi
  int v97; // edx
  __int64 v98; // rdx
  int v99; // r8d
  int v100; // edi
  int v101; // edi
  unsigned __int64 v102; // [rsp+8h] [rbp-138h]
  __int64 *v103; // [rsp+10h] [rbp-130h]
  __int64 v104; // [rsp+18h] [rbp-128h]
  __int64 v105; // [rsp+20h] [rbp-120h]
  __int64 v106; // [rsp+28h] [rbp-118h]
  __int64 *v107; // [rsp+30h] [rbp-110h]
  __int64 v108; // [rsp+30h] [rbp-110h]
  __int64 v109; // [rsp+30h] [rbp-110h]
  __int64 *v110; // [rsp+30h] [rbp-110h]
  __int64 v111; // [rsp+38h] [rbp-108h]
  __int64 v112; // [rsp+38h] [rbp-108h]
  int v113; // [rsp+38h] [rbp-108h]
  __int64 v114; // [rsp+40h] [rbp-100h]
  int v118; // [rsp+58h] [rbp-E8h]
  int v119; // [rsp+64h] [rbp-DCh] BYREF
  __int64 v120; // [rsp+68h] [rbp-D8h] BYREF
  unsigned __int64 v121; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v122; // [rsp+78h] [rbp-C8h] BYREF
  __int64 *v123; // [rsp+80h] [rbp-C0h] BYREF
  char v124; // [rsp+88h] [rbp-B8h]
  __int64 *v125[2]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 *k; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v127; // [rsp+A8h] [rbp-98h]
  __int64 v128; // [rsp+B0h] [rbp-90h]
  __int64 v129; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v130; // [rsp+C8h] [rbp-78h]
  __int64 v131; // [rsp+D0h] [rbp-70h]
  unsigned int v132; // [rsp+D8h] [rbp-68h]
  __m128i v133; // [rsp+E0h] [rbp-60h] BYREF
  _QWORD v134[10]; // [rsp+F0h] [rbp-50h] BYREF

  v129 = 0;
  v103 = (__int64 *)sub_2E311E0(a2);
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v114 = a3 + 48;
  v123 = (__int64 *)sub_2E311E0(a3);
  v102 = (unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32;
  for ( i = (__int64)v123; v114 != i; i = (__int64)v123 )
  {
    v124 = 1;
    sub_2FD79B0((__int64 *)&v123);
    if ( *(_WORD *)(i + 68) != 68 && *(_WORD *)(i + 68) )
      goto LABEL_22;
    v7 = *(unsigned int *)(a1 + 280);
    v8 = *(_QWORD *)(a1 + 264);
    if ( (_DWORD)v7 )
    {
      v9 = (v7 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( i == *v10 )
      {
LABEL_8:
        if ( v10 != (__int64 *)(v8 + 16LL * (unsigned int)v7) )
        {
          v12 = v10[1];
          goto LABEL_10;
        }
      }
      else
      {
        v52 = 1;
        while ( v11 != -4096 )
        {
          v101 = v52 + 1;
          v9 = (v7 - 1) & (v52 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( *v10 == i )
            goto LABEL_8;
          v52 = v101;
        }
      }
    }
    v12 = i;
LABEL_10:
    v13 = *(_QWORD *)a1;
    if ( a4 == (unsigned int)sub_3598DB0(*(_QWORD *)a1, v12) )
      goto LABEL_23;
    v16 = *(_QWORD *)(a1 + 24);
    LODWORD(v120) = *(_DWORD *)(*(_QWORD *)(i + 32) + 8LL);
    v17 = sub_2EC06C0(
            v16,
            *(_QWORD *)(*(_QWORD *)(v16 + 56) + 16 * (v120 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
            byte_3F871B3,
            0,
            v14,
            v15);
    v18 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
    v122 = 0;
    k = 0;
    v127 = 0;
    v128 = 0;
    v19 = (__int64 *)sub_2E311E0(a2);
    v20 = sub_2F26260(a2, v19, (__int64 *)&k, v18, v17);
    v125[1] = v21;
    v125[0] = v20;
    v22 = sub_3598AB0((__int64 *)v125, v120, 0, 0);
    v23 = v22[1];
    v133.m128i_i8[0] = 4;
    v134[0] = 0;
    v133.m128i_i32[0] &= 0xFFF000FF;
    v134[1] = a3;
    v107 = v22;
    sub_2E8EAD0(v23, *v22, &v133);
    v121 = v107[1];
    if ( k )
      sub_B91220((__int64)&k, (__int64)k);
    if ( v122 )
      sub_B91220((__int64)&v122, v122);
    v122 = i;
    v106 = v121;
    v108 = a1 + 288;
    v24 = sub_359C5E0(a1 + 256, &v122);
    v133.m128i_i64[0] = a2;
    v133.m128i_i64[1] = *v24;
    v25 = sub_359BDE0(a1 + 288, v133.m128i_i64, v125);
    v26 = a1 + 256;
    v27 = v25 == 0;
    v28 = v125[0];
    if ( !v27 )
      goto LABEL_21;
    v29 = *(_DWORD *)(a1 + 304);
    v30 = *(_DWORD *)(a1 + 312);
    k = v125[0];
    ++*(_QWORD *)(a1 + 288);
    v31 = v29 + 1;
    if ( 4 * (v29 + 1) >= 3 * v30 )
    {
      v104 = a1 + 256;
      v30 *= 2;
    }
    else
    {
      if ( v30 - *(_DWORD *)(a1 + 308) - v31 > v30 >> 3 )
        goto LABEL_18;
      v104 = a1 + 256;
    }
    sub_35A1120(v108, v30);
    sub_359BDE0(v108, v133.m128i_i64, &k);
    v26 = v104;
    v31 = *(_DWORD *)(a1 + 304) + 1;
    v28 = k;
LABEL_18:
    *(_DWORD *)(a1 + 304) = v31;
    if ( *v28 != -4096 || v28[1] != -4096 )
      --*(_DWORD *)(a1 + 308);
    *v28 = v133.m128i_i64[0];
    v32 = v133.m128i_i64[1];
    v28[2] = 0;
    v28[1] = v32;
LABEL_21:
    v109 = v26;
    v28[2] = v106;
    v133.m128i_i64[0] = i;
    v33 = *sub_359C5E0(v26, v133.m128i_i64);
    *sub_359C4A0(v109, (__int64 *)&v121) = v33;
    *sub_2FFAE70((__int64)&v129, (int *)&v120) = v17;
LABEL_22:
    v13 = *(_QWORD *)a1;
    v8 = *(_QWORD *)(a1 + 264);
    v7 = *(unsigned int *)(a1 + 280);
LABEL_23:
    if ( (_DWORD)v7 )
    {
      v34 = (v7 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v35 = (__int64 *)(v8 + 16LL * v34);
      v36 = *v35;
      if ( i == *v35 )
      {
LABEL_25:
        if ( v35 != (__int64 *)(v8 + 16 * v7) )
        {
          v37 = v35[1];
          goto LABEL_27;
        }
      }
      else
      {
        v51 = 1;
        while ( v36 != -4096 )
        {
          v100 = v51 + 1;
          v34 = (v7 - 1) & (v51 + v34);
          v35 = (__int64 *)(v8 + 16LL * v34);
          v36 = *v35;
          if ( i == *v35 )
            goto LABEL_25;
          v51 = v100;
        }
      }
    }
    v37 = i;
LABEL_27:
    if ( a4 != (unsigned int)sub_3598DB0(v13, v37) )
      continue;
    sub_2E88DB0((_QWORD *)i);
    sub_2E31040((__int64 *)(a2 + 40), i);
    v38 = *v103;
    *(_QWORD *)(i + 8) = v103;
    *(_QWORD *)i = v38 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)i & 7LL;
    *(_QWORD *)((v38 & 0xFFFFFFFFFFFFFFF8LL) + 8) = i;
    *v103 = i | *v103 & 7;
    v133.m128i_i64[0] = i;
    v39 = *sub_359C5E0(a1 + 256, v133.m128i_i64);
    v133.m128i_i64[1] = v39;
    v133.m128i_i64[0] = a2;
    v27 = (unsigned __int8)sub_359BDE0(a1 + 288, v133.m128i_i64, v125) == 0;
    v40 = v125[0];
    if ( v27 )
    {
      v41 = *(_DWORD *)(a1 + 304);
      v42 = *(_DWORD *)(a1 + 312);
      k = v125[0];
      ++*(_QWORD *)(a1 + 288);
      v43 = v41 + 1;
      if ( 4 * (v41 + 1) >= 3 * v42 )
      {
        v42 *= 2;
      }
      else if ( v42 - *(_DWORD *)(a1 + 308) - v43 > v42 >> 3 )
      {
        goto LABEL_31;
      }
      sub_35A1120(a1 + 288, v42);
      sub_359BDE0(a1 + 288, v133.m128i_i64, &k);
      v43 = *(_DWORD *)(a1 + 304) + 1;
      v40 = k;
LABEL_31:
      *(_DWORD *)(a1 + 304) = v43;
      if ( *v40 != -4096 || v40[1] != -4096 )
        --*(_DWORD *)(a1 + 308);
      *v40 = v133.m128i_i64[0];
      v44 = v133.m128i_i64[1];
      v40[2] = 0;
      v40[1] = v44;
    }
    v40[2] = i;
    v45 = *(_DWORD *)(a1 + 312);
    if ( v45 )
    {
      v46 = 1;
      v47 = v45 - 1;
      for ( j = (v45 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v102 | ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4))) >> 31)
               ^ (484763065 * (v102 | ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4)))); ; j = v47 & v50 )
      {
        v49 = (_QWORD *)(*(_QWORD *)(a1 + 296) + 24LL * j);
        if ( a3 == *v49 && v39 == v49[1] )
          break;
        if ( *v49 == -4096 && v49[1] == -4096 )
          goto LABEL_2;
        v50 = v46 + j;
        ++v46;
      }
      *v49 = -8192;
      v49[1] = -8192;
      --*(_DWORD *)(a1 + 304);
      ++*(_DWORD *)(a1 + 308);
    }
LABEL_2:
    ;
  }
  v53 = a1;
  v133.m128i_i64[0] = (__int64)v134;
  v133.m128i_i64[1] = 0x400000000LL;
  v54 = sub_2E311E0(a2);
  v55 = *(__int64 **)(a2 + 56);
  v56 = a4;
  v57 = (__int64 *)v54;
  for ( k = v55; k != v57; v55 = k )
  {
    v58 = sub_2EBEE10(*(_QWORD *)(v53 + 24), *(_DWORD *)(v55[4] + 48));
    v59 = *(_QWORD *)(v53 + 264);
    v60 = v58;
    v61 = *(unsigned int *)(v53 + 280);
    if ( (_DWORD)v61 )
    {
      v62 = (v61 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v63 = (__int64 *)(v59 + 16LL * v62);
      v64 = *v63;
      if ( v60 == *v63 )
      {
LABEL_50:
        if ( v63 != (__int64 *)(v59 + 16 * v61) )
          v60 = v63[1];
      }
      else
      {
        v93 = 1;
        while ( v64 != -4096 )
        {
          v94 = v93 + 1;
          v62 = (v61 - 1) & (v93 + v62);
          v63 = (__int64 *)(v59 + 16LL * v62);
          v64 = *v63;
          if ( v60 == *v63 )
            goto LABEL_50;
          v93 = v94;
        }
      }
    }
    if ( v56 == (unsigned int)sub_3598DB0(*(_QWORD *)v53, v60) )
    {
      v65 = v55[4];
      v118 = *(_DWORD *)(v65 + 8);
      sub_2EBECB0(*(_QWORD **)(v53 + 24), v118, *(_DWORD *)(v65 + 48));
      sub_2EAB0C0(v55[4], v118);
      v68 = v133.m128i_u32[2];
      v69 = v133.m128i_u32[2] + 1LL;
      if ( v69 > v133.m128i_u32[3] )
      {
        sub_C8D5F0((__int64)&v133, v134, v69, 8u, v66, v67);
        v68 = v133.m128i_u32[2];
      }
      *(_QWORD *)(v133.m128i_i64[0] + 8 * v68) = v55;
      ++v133.m128i_i32[2];
    }
    sub_2FD79B0((__int64 *)&k);
  }
  v70 = (__int64 *)v133.m128i_i64[0];
  v71 = v133.m128i_i64[0] + 8LL * v133.m128i_u32[2];
  if ( v71 != v133.m128i_i64[0] )
  {
    do
    {
      v72 = *v70++;
      sub_2E88E20(v72);
    }
    while ( (__int64 *)v71 != v70 );
  }
  v110 = (__int64 *)sub_2E311E0(a2);
  v120 = sub_2E311E0(a2);
  v73 = v120;
  if ( a2 + 48 != v120 )
  {
    while ( 1 )
    {
      v74 = *(_QWORD *)(v73 + 32);
      v75 = v74 + 40LL * (*(_DWORD *)(v73 + 40) & 0xFFFFFF);
      v76 = v74 + 40LL * (unsigned int)sub_2E88FE0(v73);
      if ( v75 != v76 )
        break;
LABEL_66:
      sub_2FD79B0(&v120);
      v73 = v120;
      if ( v120 == a2 + 48 )
        goto LABEL_67;
    }
    while ( 2 )
    {
      if ( !*(_BYTE *)v76 )
      {
        v77 = *(_DWORD *)(v76 + 8);
        if ( v132 )
        {
          v78 = (v132 - 1) & (37 * v77);
          v79 = (int *)(v130 + 8LL * v78);
          v80 = *v79;
          if ( v77 == *v79 )
          {
LABEL_63:
            if ( v79 != (int *)(v130 + 8LL * v132) )
            {
              sub_2EAB0C0(v76, v79[1]);
              goto LABEL_65;
            }
          }
          else
          {
            v82 = 1;
            while ( v80 != -1 )
            {
              v99 = v82 + 1;
              v78 = (v132 - 1) & (v82 + v78);
              v79 = (int *)(v130 + 8LL * v78);
              v80 = *v79;
              if ( v77 == *v79 )
                goto LABEL_63;
              v82 = v99;
            }
          }
        }
        v83 = sub_2EBEE90(*(_QWORD *)(v53 + 24), v77);
        if ( v83 && (!*(_WORD *)(v83 + 68) || *(_WORD *)(v83 + 68) == 68) && *(_QWORD *)(v83 + 24) == a3 )
        {
          v84 = *(_QWORD **)(v53 + 8);
          v121 = v83;
          v85 = (__int64)sub_2E7B2C0(v84, v83);
          v122 = v85;
          sub_2E31040((__int64 *)(a2 + 40), v85);
          v86 = *v110;
          *(_QWORD *)(v85 + 8) = v110;
          *(_QWORD *)v85 = v86 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v85 & 7LL;
          *(_QWORD *)((v86 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v85;
          *v110 = *v110 & 7 | v85;
          v87 = *(_QWORD *)(v53 + 24);
          v119 = *(_DWORD *)(*(_QWORD *)(v121 + 32) + 8LL);
          v90 = sub_2EC06C0(
                  v87,
                  *(_QWORD *)(*(_QWORD *)(v87 + 56) + 16LL * (v119 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  byte_3F871B3,
                  0,
                  v88,
                  v89);
          sub_2EAB0C0(*(_QWORD *)(v122 + 32), v90);
          sub_2EAB0C0(*(_QWORD *)(v122 + 32) + 40LL, v119);
          *(_QWORD *)(*(_QWORD *)(v122 + 32) + 104LL) = **(_QWORD **)(a2 + 64);
          *sub_2FFAE70((__int64)&v129, &v119) = v90;
          v111 = *sub_359C4A0(v53 + 256, (__int64 *)&v121);
          *sub_359C4A0(v53 + 256, &v122) = v111;
          v105 = v122;
          v112 = v53 + 288;
          v91 = sub_359C4A0(v53 + 256, (__int64 *)&v121);
          k = (__int64 *)a2;
          v127 = *v91;
          v27 = (unsigned __int8)sub_359BDE0(v53 + 288, (__int64 *)&k, &v123) == 0;
          v92 = v123;
          if ( !v27 )
            goto LABEL_77;
          v95 = *(_DWORD *)(v53 + 304);
          v96 = *(_DWORD *)(v53 + 312);
          v125[0] = v123;
          ++*(_QWORD *)(v53 + 288);
          v97 = v95 + 1;
          if ( 4 * (v95 + 1) >= 3 * v96 )
          {
            v96 *= 2;
          }
          else if ( v96 - *(_DWORD *)(v53 + 308) - v97 > v96 >> 3 )
          {
            goto LABEL_86;
          }
          sub_35A1120(v112, v96);
          sub_359BDE0(v112, (__int64 *)&k, v125);
          v97 = *(_DWORD *)(v53 + 304) + 1;
          v92 = v125[0];
LABEL_86:
          *(_DWORD *)(v53 + 304) = v97;
          if ( *v92 != -4096 || v92[1] != -4096 )
            --*(_DWORD *)(v53 + 308);
          *v92 = (__int64)k;
          v98 = v127;
          v92[2] = 0;
          v92[1] = v98;
LABEL_77:
          v92[2] = v105;
          v113 = *(_DWORD *)sub_2E263C0(v53 + 224, (__int64 *)&v121);
          *(_DWORD *)sub_2E263C0(v53 + 224, &v122) = v113;
          sub_2EAB0C0(v76, v90);
        }
      }
LABEL_65:
      v76 += 40;
      if ( v75 == v76 )
        goto LABEL_66;
      continue;
    }
  }
LABEL_67:
  if ( (_QWORD *)v133.m128i_i64[0] != v134 )
    _libc_free(v133.m128i_u64[0]);
  return sub_C7D6A0(v130, 8LL * v132, 4);
}
