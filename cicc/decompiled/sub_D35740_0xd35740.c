// Function: sub_D35740
// Address: 0xd35740
//
__int64 __fastcall sub_D35740(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // r10
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // r9
  __int64 v10; // r13
  __int64 v11; // r8
  __int64 v12; // r11
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r15
  __int64 *v18; // rax
  unsigned int v19; // r8d
  int v21; // edx
  __int64 v22; // r14
  unsigned __int64 v23; // r11
  unsigned __int64 v24; // rdx
  __int64 *v25; // rcx
  unsigned __int64 v26; // rdi
  unsigned int v27; // r8d
  __int64 *v28; // rdx
  __int64 v29; // rcx
  _DWORD *v30; // rcx
  __int64 v31; // rsi
  unsigned __int64 v32; // rdi
  unsigned int v33; // r8d
  __int64 *v34; // rdx
  __int64 v35; // r11
  _DWORD *v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r9
  _QWORD *v39; // r15
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  _BYTE *v43; // rax
  char *v44; // rdi
  __int64 v45; // r10
  _BYTE *v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rax
  char v49; // cl
  char v50; // dl
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned __int64 *v54; // rdi
  int v55; // eax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  unsigned int v62; // r15d
  int v63; // ecx
  int v64; // edx
  int v65; // r15d
  int v66; // edx
  int v67; // r15d
  _QWORD *v68; // r9
  unsigned int v69; // r12d
  _QWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // r9
  __int64 v73; // r8
  __int64 v74; // rsi
  unsigned int v75; // eax
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rt0
  __int64 v79; // r12
  __int64 v80; // r15
  unsigned int v81; // eax
  __int64 v82; // rdx
  __int64 v83; // r9
  __int64 v84; // r8
  __int64 v85; // rcx
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // r13
  unsigned __int64 v88; // rdi
  unsigned __int64 v89; // rcx
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  const __m128i *v95; // r12
  unsigned __int64 v96; // rdx
  __m128i *v97; // rax
  __int64 v98; // rdi
  char *v99; // r12
  int v100; // [rsp+Ch] [rbp-F4h]
  __int64 v101; // [rsp+18h] [rbp-E8h]
  int v102; // [rsp+18h] [rbp-E8h]
  char v103; // [rsp+20h] [rbp-E0h]
  __int64 v104; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v105; // [rsp+20h] [rbp-E0h]
  int v106; // [rsp+20h] [rbp-E0h]
  __int64 v107; // [rsp+28h] [rbp-D8h]
  _QWORD *v108; // [rsp+28h] [rbp-D8h]
  _QWORD *v109; // [rsp+28h] [rbp-D8h]
  __int64 v110; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v111; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v112; // [rsp+30h] [rbp-D0h]
  __int64 v113; // [rsp+30h] [rbp-D0h]
  char v114; // [rsp+30h] [rbp-D0h]
  int v115; // [rsp+30h] [rbp-D0h]
  unsigned int v116; // [rsp+38h] [rbp-C8h]
  _QWORD *v117; // [rsp+38h] [rbp-C8h]
  __int64 v118; // [rsp+38h] [rbp-C8h]
  unsigned __int8 v119; // [rsp+38h] [rbp-C8h]
  unsigned __int8 v120; // [rsp+38h] [rbp-C8h]
  __int64 v121; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v122; // [rsp+40h] [rbp-C0h] BYREF
  char v123; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v124; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v125; // [rsp+58h] [rbp-A8h]
  int v126; // [rsp+60h] [rbp-A0h]
  char v127; // [rsp+64h] [rbp-9Ch]
  char *v128; // [rsp+70h] [rbp-90h] BYREF
  char v129; // [rsp+80h] [rbp-80h] BYREF
  char *v130; // [rsp+A0h] [rbp-60h] BYREF
  char v131; // [rsp+B0h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_QWORD *)(a1 + 280);
  v7 = **(unsigned int **)(a2 + 16);
  v8 = *(_DWORD *)(v6 + 48);
  v9 = *(_QWORD *)(v6 + 32);
  v10 = v5 + 72 * v7;
  v11 = *(unsigned __int8 *)(v10 + 40);
  v12 = **(unsigned int **)(a3 + 16);
  v13 = *(_QWORD *)(v10 + 16) & 0xFFFFFFFFFFFFFFFBLL;
  v14 = v13 | (4LL * (*(_BYTE *)(v10 + 40) ^ 1u));
  if ( v8 )
  {
    v116 = v8 - 1;
    v15 = (v8 - 1) & (v14 ^ (v14 >> 9));
    v16 = (__int64 *)(v9 + 32LL * v15);
    v17 = *v16;
    if ( v14 == *v16 )
    {
LABEL_3:
      v18 = (__int64 *)(v9 + 32LL * v8);
      if ( v18 == v16 )
      {
        v22 = 72LL * (unsigned int)v12 + v5;
      }
      else
      {
        if ( v16[1] != v16[2] )
          return 0;
        v22 = v5 + 72 * v12;
      }
      v112 = *(_BYTE *)(v22 + 40);
      v23 = *(_QWORD *)(v22 + 16) & 0xFFFFFFFFFFFFFFFBLL;
      v24 = v23 | (4LL * (v112 ^ 1u));
      goto LABEL_11;
    }
    v21 = 1;
    while ( v17 != -4 )
    {
      v15 = v116 & (v21 + v15);
      v115 = v21 + 1;
      v16 = (__int64 *)(v9 + 32LL * v15);
      v17 = *v16;
      if ( v14 == *v16 )
        goto LABEL_3;
      v21 = v115;
    }
  }
  v22 = v5 + 72 * v12;
  v18 = (__int64 *)(v9 + 32LL * v8);
  v112 = *(_BYTE *)(v22 + 40);
  v23 = *(_QWORD *)(v22 + 16) & 0xFFFFFFFFFFFFFFFBLL;
  v24 = v23 | (4LL * (v112 ^ 1u));
  if ( !v8 )
    goto LABEL_45;
  v116 = v8 - 1;
LABEL_11:
  v25 = (__int64 *)(v9 + 32LL * (v116 & ((unsigned int)v24 ^ (unsigned int)(v24 >> 9))));
  v107 = *v25;
  if ( *v25 != v24 )
  {
    v62 = v116 & (v24 ^ (v24 >> 9));
    v63 = 1;
    while ( v107 != -4 )
    {
      v62 = v116 & (v63 + v62);
      v102 = v63 + 1;
      v25 = (__int64 *)(v9 + 32LL * v62);
      v107 = *v25;
      if ( v24 == *v25 )
        goto LABEL_12;
      v63 = v102;
    }
    goto LABEL_45;
  }
LABEL_12:
  if ( v18 == v25 )
  {
LABEL_45:
    v26 = (4 * v11) | v13;
    if ( !v8 )
      return 0;
    v116 = v8 - 1;
    goto LABEL_15;
  }
  if ( v25[1] != v25[2] )
    return 0;
  v26 = (4 * v11) | v13;
LABEL_15:
  v27 = v116 & (v26 ^ (v26 >> 9));
  v28 = (__int64 *)(v9 + 32LL * v27);
  v29 = *v28;
  if ( v26 == *v28 )
  {
LABEL_16:
    if ( v18 != v28 )
    {
      v30 = (_DWORD *)v28[1];
      v31 = (v28[2] - (__int64)v30) >> 2;
      v32 = v23 | (4LL * v112);
      goto LABEL_18;
    }
    v32 = v23 | (4LL * v112);
  }
  else
  {
    v66 = 1;
    while ( v29 != -4 )
    {
      v27 = v116 & (v66 + v27);
      v67 = v66 + 1;
      v28 = (__int64 *)(v9 + 32LL * v27);
      v29 = *v28;
      if ( v26 == *v28 )
        goto LABEL_16;
      v66 = v67;
    }
    v32 = v23 | (4LL * v112);
  }
  v30 = 0;
  v31 = 0;
LABEL_18:
  v33 = v116 & (v32 ^ (v32 >> 9));
  v34 = (__int64 *)(v9 + 32LL * v33);
  v35 = *v34;
  if ( v32 != *v34 )
  {
    v64 = 1;
    while ( v35 != -4 )
    {
      v33 = v116 & (v64 + v33);
      v65 = v64 + 1;
      v34 = (__int64 *)(v9 + 32LL * v33);
      v35 = *v34;
      if ( v32 == *v34 )
        goto LABEL_19;
      v64 = v65;
    }
    return 0;
  }
LABEL_19:
  if ( v34 == v18 )
    return 0;
  v36 = (_DWORD *)v34[1];
  v19 = 0;
  if ( v31 == 1 && v34[2] - (_QWORD)v36 == 4 )
  {
    if ( *v36 < *v30 )
    {
      v37 = v10;
      v10 = v22;
      v22 = v37;
    }
    v38 = *(_QWORD *)(v10 + 56);
    v19 = 0;
    v39 = *(_QWORD **)(v22 + 56);
    if ( *(_WORD *)(v38 + 24) == 8 && *((_WORD *)v39 + 12) == 8 )
    {
      v40 = *(_QWORD *)(v6 + 8);
      if ( *(_QWORD *)(v38 + 48) == v40 && v40 == v39[6] )
      {
        v117 = *(_QWORD **)(v10 + 56);
        sub_D35600((__int64)&v128, v6, *(_QWORD *)(v10 + 16), *(_BYTE *)(v10 + 40), 0, v38);
        sub_D35600((__int64)&v130, *(_QWORD *)(a1 + 280), *(_QWORD *)(v22 + 16), *(_BYTE *)(v22 + 40), v41, v42);
        v43 = *(_BYTE **)v128;
        if ( **(_BYTE **)v128 != 61 )
          v43 = (_BYTE *)*((_QWORD *)v43 - 8);
        v44 = v130;
        v45 = *((_QWORD *)v43 + 1);
        v46 = *(_BYTE **)v130;
        if ( **(_BYTE **)v130 != 61 )
          v46 = (_BYTE *)*((_QWORD *)v46 - 8);
        v19 = 0;
        v47 = *((_QWORD *)v46 + 1);
        v108 = v117;
        v113 = v45;
        if ( *(_BYTE *)(v45 + 8) == 18 || *(_BYTE *)(v47 + 8) == 18 )
          goto LABEL_39;
        v118 = sub_AA4E30(**(_QWORD **)(v39[6] + 32LL));
        v103 = sub_AE5020(v118, v47);
        v48 = sub_9208B0(v118, v47);
        v49 = v103;
        LOBYTE(v125) = v50;
        v51 = 1LL << v103;
        v104 = v113;
        v122 = (((unsigned __int64)(v48 + 7) >> 3) + v51 - 1) >> v49 << v49;
        v123 = v125;
        v114 = sub_AE5020(v118, v113);
        v52 = sub_9208B0(v118, v104);
        v125 = v53;
        v124 = (((unsigned __int64)(v52 + 7) >> 3) + (1LL << v114) - 1) >> v114 << v114;
        v105 = sub_CA1930(&v124);
        v54 = &v122;
        if ( v105 >= sub_CA1930(&v122) )
          v54 = &v124;
        v55 = sub_CA1930(v54);
        v47 = *(_QWORD *)(a1 + 288);
        v106 = v55;
        v101 = sub_D33D80(v39, v47, v56, v57, v58);
        if ( !*(_WORD *)(v101 + 24) )
        {
          v47 = *(_QWORD *)(a1 + 288);
          if ( v101 == sub_D33D80(v108, v47, v59, v60, v61) )
          {
            v47 = *(_QWORD *)(v101 + 32) + 24LL;
            sub_9692E0((__int64)&v124, (__int64 *)v47);
            v68 = v108;
            v19 = 0;
            v100 = v125;
            if ( (unsigned int)v125 > 0x40 )
            {
              if ( v100 - (unsigned int)sub_C444A0((__int64)&v124) > 0x40 || v106 != *(_QWORD *)v124 )
              {
                if ( v124 )
                {
                  j_j___libc_free_0_0(v124);
                  v44 = v130;
                  v19 = 0;
                  goto LABEL_39;
                }
                goto LABEL_38;
              }
              j_j___libc_free_0_0(v124);
              v68 = v108;
            }
            else if ( v106 != v124 )
            {
              v44 = v130;
              goto LABEL_39;
            }
            v109 = v68;
            v69 = sub_AE2980(v118, *(_DWORD *)(a2 + 40))[1];
            v70 = (_QWORD *)sub_BD5C60(*(_QWORD *)(v10 + 16));
            v71 = sub_BCCE00(v70, v69);
            v72 = (__int64)v109;
            v73 = v71;
            v74 = *(_QWORD *)(v101 + 32);
            v75 = *(_DWORD *)(v74 + 32);
            v76 = 1LL << ((unsigned __int8)v75 - 1);
            if ( v75 > 0x40 )
              v77 = *(_QWORD *)(*(_QWORD *)(v74 + 24) + 8LL * ((v75 - 1) >> 6));
            else
              v77 = *(_QWORD *)(v74 + 24);
            if ( (v77 & v76) != 0 )
            {
              v78 = (__int64)v39;
              v39 = v109;
              v72 = v78;
            }
            v121 = v72;
            v110 = v73;
            v79 = sub_DD3A70(*(_QWORD *)(a1 + 288), *(_QWORD *)v39[4], v73);
            v47 = **(_QWORD **)(v121 + 32);
            v80 = sub_DD3A70(*(_QWORD *)(a1 + 288), v47, v110);
            if ( !(unsigned __int8)sub_D96A50(v79) )
            {
              v81 = sub_D96A50(v80);
              v83 = v121;
              v84 = v81;
              if ( !(_BYTE)v81 )
              {
                if ( *(_BYTE *)qword_4F86DC8
                  && (v85 = **(_QWORD **)(v121 + 48)) != 0
                  && *(_WORD *)(v79 + 24) == 8
                  && *(_WORD *)(v80 + 24) == 8
                  && (v90 = *(_QWORD *)(v80 + 48), *(_QWORD *)(v79 + 48) == v90)
                  && v85 == v90
                  && (v111 = v84,
                      v91 = sub_D33D80((_QWORD *)v80, *(_QWORD *)(a1 + 288), v82, v85, v84),
                      v47 = *(_QWORD *)(a1 + 288),
                      v91 != sub_D33D80((_QWORD *)v79, v47, v92, v93, v94)) )
                {
                  v44 = v130;
                  v19 = v111;
                }
                else
                {
                  v47 = *(unsigned __int8 *)(v10 + 64);
                  if ( !(_BYTE)v47 )
                    v47 = *(unsigned __int8 *)(v22 + 64);
                  v86 = *(unsigned int *)(a1 + 392);
                  v87 = *(_QWORD *)(a1 + 384);
                  v88 = *(unsigned int *)(a1 + 396);
                  v89 = v87 + 24 * v86;
                  if ( v86 >= v88 )
                  {
                    v125 = v79;
                    v124 = v80;
                    v95 = (const __m128i *)&v124;
                    v126 = v106;
                    v96 = v86 + 1;
                    v127 = v47;
                    if ( v88 < v86 + 1 )
                    {
                      v98 = a1 + 384;
                      v47 = a1 + 400;
                      if ( v87 > (unsigned __int64)&v124 || v89 <= (unsigned __int64)&v124 )
                      {
                        sub_C8D5F0(v98, (const void *)v47, v96, 0x18u, v84, v83);
                        v95 = (const __m128i *)&v124;
                        v86 = *(unsigned int *)(a1 + 392);
                        v87 = *(_QWORD *)(a1 + 384);
                      }
                      else
                      {
                        v99 = (char *)&v124 - v87;
                        sub_C8D5F0(v98, (const void *)v47, v96, 0x18u, v84, v83);
                        v87 = *(_QWORD *)(a1 + 384);
                        v86 = *(unsigned int *)(a1 + 392);
                        v95 = (const __m128i *)&v99[v87];
                      }
                    }
                    v19 = 1;
                    v97 = (__m128i *)(v87 + 24 * v86);
                    *v97 = _mm_loadu_si128(v95);
                    v44 = v130;
                    v97[1].m128i_i64[0] = v95[1].m128i_i64[0];
                    ++*(_DWORD *)(a1 + 392);
                  }
                  else
                  {
                    if ( v89 )
                    {
                      *(_QWORD *)v89 = v80;
                      *(_QWORD *)(v89 + 8) = v79;
                      *(_DWORD *)(v89 + 16) = v106;
                      *(_BYTE *)(v89 + 20) = v47;
                    }
                    ++*(_DWORD *)(a1 + 392);
                    v44 = v130;
                    v19 = 1;
                  }
                }
                goto LABEL_39;
              }
            }
          }
        }
LABEL_38:
        v44 = v130;
        v19 = 0;
LABEL_39:
        if ( v44 != &v131 )
        {
          v119 = v19;
          _libc_free(v44, v47);
          v19 = v119;
        }
        if ( v128 != &v129 )
        {
          v120 = v19;
          _libc_free(v128, v47);
          return v120;
        }
      }
    }
  }
  return v19;
}
