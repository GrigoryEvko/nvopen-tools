// Function: sub_2F6A580
// Address: 0x2f6a580
//
__int16 __fastcall sub_2F6A580(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  char v4; // cl
  __int64 v5; // r14
  int v6; // r15d
  unsigned __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // r12
  int v10; // r15d
  unsigned int v11; // edx
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  __int64 v14; // r8
  unsigned __int64 i; // rcx
  __int64 j; // rsi
  __int16 v17; // dx
  __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // r9
  unsigned __int64 v23; // r15
  __int64 v24; // rbx
  __int64 *v25; // rdx
  __int64 v26; // r14
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int16 result; // ax
  __int64 v30; // r9
  unsigned int v31; // esi
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // r10
  __int64 v35; // r9
  __int64 *v36; // r8
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 *v39; // rax
  __int64 *v40; // rdi
  unsigned int v41; // edx
  __int64 v42; // rcx
  __int64 *v43; // rbx
  __int64 v44; // rax
  unsigned int v45; // edx
  __int64 v46; // rcx
  __int64 *v47; // rbx
  __int64 v48; // rax
  int v49; // edx
  __int64 v50; // r13
  __int64 v51; // r8
  _QWORD *v52; // rdx
  _QWORD *v53; // rsi
  __int64 v54; // r12
  __int64 v55; // r8
  _QWORD *v56; // rdx
  _QWORD *v57; // rsi
  int v58; // r10d
  unsigned int v59; // ecx
  unsigned int v60; // edx
  unsigned int v61; // esi
  __int64 v62; // rax
  __int64 v63; // r9
  __int64 v64; // r10
  __int64 v65; // rcx
  __int64 v66; // r11
  __int64 v67; // rbx
  __int64 v68; // r13
  unsigned __int64 v69; // r8
  __int64 v70; // r15
  __int64 v71; // r10
  unsigned __int64 v72; // r12
  __int64 v73; // r14
  __int64 *v74; // rax
  __int64 v75; // rax
  __int64 v76; // r9
  __int64 v77; // r10
  __int64 v78; // r8
  int v79; // eax
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // r9
  __int64 v84; // r10
  __int64 v85; // r8
  __int64 v86; // r14
  __int64 v87; // r15
  bool v88; // zf
  __int64 v89; // rdx
  __int64 v90; // rax
  _QWORD *v91; // r12
  _QWORD *v92; // rcx
  _QWORD *v93; // rax
  __int64 v94; // r12
  __int64 v95; // r14
  __int64 v96; // r13
  __int128 v97; // rax
  const __m128i *v98; // rdx
  unsigned __int64 v99; // rsi
  __int64 v100; // rax
  unsigned __int64 v101; // rbx
  __int64 *v102; // rcx
  __int64 v103; // rax
  _DWORD *v104; // rax
  __int64 v105; // rsi
  __int64 v106; // rax
  __int64 *v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // r8
  __int64 v110; // r9
  __int64 v111; // rbx
  __int64 v112; // rax
  unsigned __int64 v113; // rax
  int v114; // esi
  __int64 v115; // rax
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // [rsp+0h] [rbp-C0h]
  __int64 v121; // [rsp+0h] [rbp-C0h]
  __int64 v122; // [rsp+8h] [rbp-B8h]
  __int64 v123; // [rsp+8h] [rbp-B8h]
  _QWORD *v124; // [rsp+10h] [rbp-B0h]
  __int64 v125; // [rsp+18h] [rbp-A8h]
  __int64 v126; // [rsp+18h] [rbp-A8h]
  __int64 v127; // [rsp+18h] [rbp-A8h]
  __int64 v128; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v129; // [rsp+20h] [rbp-A0h]
  __int64 v130; // [rsp+20h] [rbp-A0h]
  __int64 v131; // [rsp+20h] [rbp-A0h]
  __int64 v132; // [rsp+20h] [rbp-A0h]
  __int64 v133; // [rsp+28h] [rbp-98h]
  __int64 v134; // [rsp+28h] [rbp-98h]
  _BYTE *v135; // [rsp+28h] [rbp-98h]
  unsigned __int64 v136; // [rsp+28h] [rbp-98h]
  __int64 v137; // [rsp+28h] [rbp-98h]
  __int64 v138; // [rsp+30h] [rbp-90h]
  __int64 v139; // [rsp+30h] [rbp-90h]
  __int64 v140; // [rsp+30h] [rbp-90h]
  __int64 v141; // [rsp+30h] [rbp-90h]
  __int64 v142; // [rsp+30h] [rbp-90h]
  __int64 v143; // [rsp+30h] [rbp-90h]
  __int64 v144; // [rsp+30h] [rbp-90h]
  __int64 v145; // [rsp+38h] [rbp-88h]
  __int64 *v146; // [rsp+38h] [rbp-88h]
  __int64 v147; // [rsp+38h] [rbp-88h]
  __int64 v148; // [rsp+38h] [rbp-88h]
  __int64 v149; // [rsp+38h] [rbp-88h]
  __int64 v150; // [rsp+38h] [rbp-88h]
  __int64 v151; // [rsp+38h] [rbp-88h]
  __int64 v152; // [rsp+38h] [rbp-88h]
  __int64 v153; // [rsp+40h] [rbp-80h]
  unsigned int v154; // [rsp+40h] [rbp-80h]
  __int64 v155; // [rsp+40h] [rbp-80h]
  __int64 v156; // [rsp+40h] [rbp-80h]
  __int64 v157; // [rsp+40h] [rbp-80h]
  __int64 v158; // [rsp+48h] [rbp-78h]
  __int64 v159; // [rsp+48h] [rbp-78h]
  __int64 v160; // [rsp+48h] [rbp-78h]
  __int64 *v162; // [rsp+50h] [rbp-70h]
  __int64 v163; // [rsp+50h] [rbp-70h]
  __int64 v164; // [rsp+50h] [rbp-70h]
  char v166; // [rsp+67h] [rbp-59h] BYREF
  unsigned int v167; // [rsp+68h] [rbp-58h] BYREF
  unsigned int v168; // [rsp+6Ch] [rbp-54h] BYREF
  __m128i v169; // [rsp+70h] [rbp-50h] BYREF
  __int64 (__fastcall *v170)(unsigned __int64 *, const __m128i **, int); // [rsp+80h] [rbp-40h]
  __int64 (__fastcall *v171)(__int64 *, __int64); // [rsp+88h] [rbp-38h]

  v4 = *(_BYTE *)(a2 + 26);
  v5 = *(_QWORD *)(a1 + 40);
  if ( v4 )
    v6 = *(_DWORD *)(a2 + 8);
  else
    v6 = *(_DWORD *)(a2 + 12);
  v7 = *(unsigned int *)(v5 + 160);
  v8 = v6 & 0x7FFFFFFF;
  if ( (v6 & 0x7FFFFFFFu) >= (unsigned int)v7 || (v9 = *(_QWORD *)(*(_QWORD *)(v5 + 152) + 8LL * v8)) == 0 )
  {
    v45 = v8 + 1;
    if ( (unsigned int)v7 < v45 && v45 != v7 )
    {
      if ( v45 >= v7 )
      {
        v54 = *(_QWORD *)(v5 + 168);
        v55 = v45 - v7;
        if ( v45 > (unsigned __int64)*(unsigned int *)(v5 + 164) )
        {
          v160 = v45 - v7;
          sub_C8D5F0(v5 + 152, (const void *)(v5 + 168), v45, 8u, v55, v45);
          v7 = *(unsigned int *)(v5 + 160);
          v55 = v160;
        }
        v46 = *(_QWORD *)(v5 + 152);
        v56 = (_QWORD *)(v46 + 8 * v7);
        v57 = &v56[v55];
        if ( v56 != v57 )
        {
          do
            *v56++ = v54;
          while ( v57 != v56 );
          LODWORD(v7) = *(_DWORD *)(v5 + 160);
          v46 = *(_QWORD *)(v5 + 152);
        }
        *(_DWORD *)(v5 + 160) = v55 + v7;
LABEL_51:
        v47 = (__int64 *)(v46 + 8LL * (v6 & 0x7FFFFFFF));
        v48 = sub_2E10F30(v6);
        *v47 = v48;
        v9 = v48;
        sub_2E11E80((_QWORD *)v5, v48);
        v5 = *(_QWORD *)(a1 + 40);
        v7 = *(unsigned int *)(v5 + 160);
        if ( *(_BYTE *)(a2 + 26) )
          goto LABEL_6;
LABEL_52:
        v10 = *(_DWORD *)(a2 + 8);
        goto LABEL_7;
      }
      *(_DWORD *)(v5 + 160) = v45;
    }
    v46 = *(_QWORD *)(v5 + 152);
    goto LABEL_51;
  }
  if ( !v4 )
    goto LABEL_52;
LABEL_6:
  v10 = *(_DWORD *)(a2 + 12);
LABEL_7:
  v11 = v10 & 0x7FFFFFFF;
  if ( (v10 & 0x7FFFFFFFu) >= (unsigned int)v7 || (v12 = *(_QWORD *)(*(_QWORD *)(v5 + 152) + 8LL * v11)) == 0 )
  {
    v41 = v11 + 1;
    if ( v41 > (unsigned int)v7 && v41 != v7 )
    {
      if ( v41 >= v7 )
      {
        v50 = *(_QWORD *)(v5 + 168);
        v51 = v41 - v7;
        if ( v41 > (unsigned __int64)*(unsigned int *)(v5 + 164) )
        {
          v159 = v41 - v7;
          sub_C8D5F0(v5 + 152, (const void *)(v5 + 168), v41, 8u, v51, v41);
          v7 = *(unsigned int *)(v5 + 160);
          v51 = v159;
        }
        v42 = *(_QWORD *)(v5 + 152);
        v52 = (_QWORD *)(v42 + 8 * v7);
        v53 = &v52[v51];
        if ( v52 != v53 )
        {
          do
            *v52++ = v50;
          while ( v53 != v52 );
          LODWORD(v7) = *(_DWORD *)(v5 + 160);
          v42 = *(_QWORD *)(v5 + 152);
        }
        *(_DWORD *)(v5 + 160) = v51 + v7;
        goto LABEL_48;
      }
      *(_DWORD *)(v5 + 160) = v41;
    }
    v42 = *(_QWORD *)(v5 + 152);
LABEL_48:
    v43 = (__int64 *)(v42 + 8LL * (v10 & 0x7FFFFFFF));
    v44 = sub_2E10F30(v10);
    *v43 = v44;
    v12 = v44;
    sub_2E11E80((_QWORD *)v5, v44);
    v5 = *(_QWORD *)(a1 + 40);
  }
  v13 = a3;
  v14 = *(_QWORD *)(v5 + 32);
  for ( i = a3; (*(_BYTE *)(v13 + 44) & 4) != 0; v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  if ( (*(_DWORD *)(a3 + 44) & 8) != 0 )
  {
    do
      i = *(_QWORD *)(i + 8);
    while ( (*(_BYTE *)(i + 44) & 8) != 0 );
  }
  for ( j = *(_QWORD *)(i + 8); j != v13; v13 = *(_QWORD *)(v13 + 8) )
  {
    v17 = *(_WORD *)(v13 + 68);
    if ( (unsigned __int16)(v17 - 14) > 4u && v17 != 24 )
      break;
  }
  v18 = *(unsigned int *)(v14 + 144);
  v19 = *(_QWORD *)(v14 + 128);
  if ( (_DWORD)v18 )
  {
    v20 = (v18 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v21 = (__int64 *)(v19 + 16LL * v20);
    v22 = *v21;
    if ( v13 == *v21 )
      goto LABEL_19;
    v49 = 1;
    while ( v22 != -4096 )
    {
      v58 = v49 + 1;
      v20 = (v18 - 1) & (v49 + v20);
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == v13 )
        goto LABEL_19;
      v49 = v58;
    }
  }
  v21 = (__int64 *)(v19 + 16 * v18);
LABEL_19:
  v23 = v21[1] & 0xFFFFFFFFFFFFFFF8LL;
  v24 = v23 | 4;
  v25 = (__int64 *)sub_2E09D00((__int64 *)v12, v23 | 4);
  if ( v25 == (__int64 *)(*(_QWORD *)v12 + 24LL * *(unsigned int *)(v12 + 8))
    || (*(_DWORD *)((*v25 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v25 >> 1) & 3) > (*(_DWORD *)(v23 + 24) | 2u) )
  {
    v158 = 0;
  }
  else
  {
    v158 = v25[2];
  }
  v26 = v23 | 2;
  v27 = (__int64 *)sub_2E09D00((__int64 *)v9, v23 | 2);
  if ( v27 == (__int64 *)(*(_QWORD *)v9 + 24LL * *(unsigned int *)(v9 + 8))
    || (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) > (*(_DWORD *)(v23 + 24) | 1u) )
  {
    BUG();
  }
  v28 = *(_QWORD *)(v27[2] + 8);
  if ( (v28 & 6) == 0 )
    return 0;
  v30 = *(_QWORD *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( !v30 )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)(v30 + 16) + 27LL) & 2) == 0 )
    return 0;
  v153 = *(_QWORD *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  v145 = v27[2];
  v31 = sub_2E8E710(v30, *(_DWORD *)(v9 + 112), 0, 0, 0);
  v32 = *(_QWORD *)(v153 + 32) + 40LL * v31;
  if ( *(_BYTE *)v32 )
    return 0;
  if ( (*(_BYTE *)(v32 + 3) & 0x10) == 0 )
    return 0;
  if ( (*(_WORD *)(v32 + 2) & 0xFF0) == 0 )
    return 0;
  v168 = -1;
  v167 = sub_2E89F40(v153, v31);
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, unsigned int *, unsigned int *))(**(_QWORD **)(a1 + 32)
                                                                                           + 264LL))(
          *(_QWORD *)(a1 + 32),
          v153,
          &v167,
          &v168) )
    return 0;
  v33 = v153;
  v154 = *(_DWORD *)(*(_QWORD *)(v153 + 32) + 40LL * v168 + 8);
  if ( *(_DWORD *)(v12 + 112) != v154 )
    return 0;
  v138 = v33;
  sub_2F65960((__int64)&v169, v12, *(_QWORD *)(v145 + 8));
  if ( !(_BYTE)v171 || (unsigned __int8)sub_2E13670(*(_QWORD *)(a1 + 40), v9, v145) )
    return 0;
  v34 = v145;
  v35 = v138;
  if ( *(_QWORD *)v9 != *(_QWORD *)v9 + 24LL * *(unsigned int *)(v9 + 8) )
  {
    v36 = *(__int64 **)v9;
    do
    {
      if ( v34 == v36[2] )
      {
        v37 = 24LL * *(unsigned int *)(v12 + 8);
        v146 = (__int64 *)(*(_QWORD *)v12 + v37);
        v38 = 0xAAAAAAAAAAAAAAABLL * (v37 >> 3);
        if ( v37 )
        {
          v39 = *(__int64 **)v12;
          do
          {
            v40 = &v39[(v38 >> 1) + (v38 & 0xFFFFFFFFFFFFFFFELL)];
            if ( (*(_DWORD *)((*v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v36 >> 1) & 3) >= (*(_DWORD *)((*v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v40 >> 1) & 3) )
            {
              v39 = v40 + 3;
              v38 = v38 - (v38 >> 1) - 1;
            }
            else
            {
              v38 >>= 1;
            }
          }
          while ( v38 > 0 );
          if ( *(__int64 **)v12 != v39 )
            v39 -= 3;
        }
        else
        {
          v39 = *(__int64 **)v12;
        }
        if ( v39 != v146 )
        {
          v59 = *(_DWORD *)((v36[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v36[1] >> 1) & 3;
          do
          {
            v60 = *(_DWORD *)((*v39 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v39 >> 1) & 3;
            if ( v59 < v60 )
              break;
            if ( v158 != v39[2] )
            {
              v61 = *(_DWORD *)((*v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v36 >> 1) & 3;
              if ( v60 <= v61 )
              {
                if ( v61 < (*(_DWORD *)((v39[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v39[1] >> 1) & 3) )
                  return 0;
              }
              else if ( v59 > v60 )
              {
                return 0;
              }
            }
            v39 += 3;
          }
          while ( v39 != v146 );
        }
      }
      v36 += 3;
    }
    while ( (__int64 *)(*(_QWORD *)v9 + 24LL * *(unsigned int *)(v9 + 8)) != v36 );
    v35 = v138;
  }
  v139 = v34;
  v147 = v35;
  v62 = sub_2F657C0(*(_QWORD *)(a1 + 16), *(_DWORD *)(v9 + 112));
  v63 = v147;
  v64 = v139;
  if ( v62 )
  {
    v65 = v23 | 4;
    v66 = v12;
    v67 = v62;
    v68 = v9;
    v69 = v23;
    v70 = v139;
    v71 = v26;
LABEL_92:
    v72 = *(_QWORD *)(v67 + 16);
    v122 = v66;
    v125 = v71;
    v129 = v69;
    v133 = v65;
    v73 = *(_QWORD *)(v72 + 32);
    v140 = v63;
    v148 = sub_2DF8360(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v72, 0);
    v74 = (__int64 *)sub_2E09D00((__int64 *)v68, v148);
    v63 = v140;
    v65 = v133;
    v69 = v129;
    v71 = v125;
    v66 = v122;
    if ( v74 != (__int64 *)(*(_QWORD *)v68 + 24LL * *(unsigned int *)(v68 + 8))
      && (*(_DWORD *)((*v74 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v74 >> 1) & 3) <= (*(_DWORD *)((v148 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v148 >> 1)
                                                                                             & 3)
      && v74[2] == v70 )
    {
      v113 = *(_QWORD *)(v72 + 32) + 0xFFFFFFF800000008LL * (unsigned int)((v67 - v73) >> 3);
      if ( !*(_BYTE *)v113 && (*(_BYTE *)(v113 + 3) & 0x10) == 0 && (*(_WORD *)(v113 + 2) & 0xFF0) != 0 )
        return 0;
    }
    while ( 1 )
    {
      v67 = *(_QWORD *)(v67 + 32);
      if ( !v67 )
        break;
      if ( (*(_BYTE *)(v67 + 3) & 0x10) == 0 && (*(_BYTE *)(v67 + 4) & 8) == 0 )
        goto LABEL_92;
    }
    v26 = v125;
    v9 = v68;
    v64 = v70;
    v24 = v133;
    v23 = v129;
    v12 = v122;
  }
  v134 = v64;
  v149 = *(_QWORD *)(v63 + 24);
  v141 = v63;
  v75 = sub_2FDF330(*(_QWORD *)(a1 + 32), v63, 0, v167, v168);
  v76 = v141;
  v77 = v134;
  v78 = v75;
  if ( !v75 )
    return 0;
  v79 = *(_DWORD *)(v9 + 112);
  if ( v79 < 0 )
  {
    v114 = *(_DWORD *)(v12 + 112);
    if ( v114 < 0 )
    {
      v132 = v134;
      v137 = v78;
      v115 = sub_2EBE590(
               *(_QWORD *)(a1 + 16),
               v114,
               *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 56LL) + 16LL * (v79 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
               0);
      v76 = v141;
      v78 = v137;
      v77 = v132;
      if ( !v115 )
        return 0;
    }
  }
  if ( v76 != v78 )
  {
    v130 = v77;
    v135 = (_BYTE *)v76;
    v142 = v78;
    sub_2F6A220(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), v76, v78);
    sub_2E31040((__int64 *)(v149 + 40), v142);
    v80 = *(_QWORD *)v135;
    v81 = *(_QWORD *)v142;
    *(_QWORD *)(v142 + 8) = v135;
    v80 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v142 = v80 | v81 & 7;
    *(_QWORD *)(v80 + 8) = v142;
    *(_QWORD *)v135 = v142 | *(_QWORD *)v135 & 7LL;
    sub_2F65590(v149, v135);
    v77 = v130;
  }
  v150 = v77;
  v82 = sub_2F65770(*(_QWORD *)(a1 + 16), *(_DWORD *)(v9 + 112));
  v84 = v150;
  v85 = v82;
  if ( v82 )
  {
    v151 = v24;
    v136 = v23;
    v131 = v26;
    v143 = v84;
    while ( 1 )
    {
      v86 = *(_QWORD *)(v85 + 32);
      if ( v86 )
      {
        do
        {
          if ( (*(_BYTE *)(v86 + 3) & 0x10) == 0 )
            break;
          v86 = *(_QWORD *)(v86 + 32);
        }
        while ( v86 );
        if ( (*(_BYTE *)(v85 + 4) & 1) != 0 )
          goto LABEL_113;
      }
      else if ( (*(_BYTE *)(v85 + 4) & 1) != 0 )
      {
LABEL_115:
        v24 = v151;
        v23 = v136;
        v26 = v131;
        v84 = v143;
        break;
      }
      v87 = *(_QWORD *)(v85 + 16);
      if ( (unsigned __int16)(*(_WORD *)(v87 + 68) - 14) > 4u )
      {
        v127 = v85;
        v101 = sub_2DF8360(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), *(_QWORD *)(v85 + 16), 0) & 0xFFFFFFFFFFFFFFF8LL;
        v102 = (__int64 *)sub_2E09D00((__int64 *)v9, v101 | 2);
        v103 = *(_QWORD *)v9 + 24LL * *(unsigned int *)(v9 + 8);
        if ( v102 != (__int64 *)v103
          && (*(_DWORD *)((*v102 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v102 >> 1) & 3) <= (*(_DWORD *)(v101 + 24) | 1u) )
        {
          v103 = (__int64)v102;
        }
        if ( *(_QWORD *)(v103 + 16) == v143 )
        {
          *(_BYTE *)(v127 + 3) &= ~0x40u;
          if ( v154 - 1 > 0x3FFFFFFE )
            sub_2EAB0C0(v127, v154);
          else
            sub_2EAB1E0(v127, v154, *(_QWORD **)(a1 + 24));
          if ( a3 != v87 && *(_WORD *)(v87 + 68) == 20 )
          {
            v104 = *(_DWORD **)(v87 + 32);
            if ( *(_DWORD *)(v12 + 112) == v104[2] && (*v104 & 0xFFF00) == 0 )
            {
              v128 = v101 | 4;
              v105 = sub_2F658F0(v12, v101 | 4);
              if ( v105 )
              {
                v106 = sub_2E0AAF0(v12, v105, v158);
                v111 = *(_QWORD *)(v12 + 104);
                v158 = v106;
                while ( v111 )
                {
                  v121 = sub_2F658F0(v111, v128);
                  if ( v121 )
                  {
                    v112 = sub_2F658F0(v111, v151);
                    sub_2E0AAF0(v111, v121, v112);
                  }
                  v111 = *(_QWORD *)(v111 + 104);
                }
                sub_2F626D0(a1, v87, v107, v108, v109, v110);
                v85 = v86;
                goto LABEL_114;
              }
            }
          }
        }
      }
      else
      {
        sub_2EAB0C0(v85, v154);
      }
LABEL_113:
      v85 = v86;
LABEL_114:
      if ( !v86 )
        goto LABEL_115;
    }
  }
  v88 = *(_QWORD *)(v9 + 104) == 0;
  v166 = 0;
  v89 = *(_QWORD *)(v12 + 104);
  v90 = *(_QWORD *)(a1 + 40);
  v162 = (__int64 *)(v90 + 56);
  if ( !v88 )
  {
    if ( !v89 )
    {
      v156 = v84;
      v116 = sub_2EBF1E0(*(_QWORD *)(a1 + 16), *(_DWORD *)(v12 + 112));
      sub_2F687C0(v12, v162, v116, v117, (__int64 *)v12);
      v84 = v156;
      v90 = *(_QWORD *)(a1 + 40);
    }
    goto LABEL_119;
  }
  if ( v89 )
  {
    v157 = v84;
    v118 = sub_2EBF1E0(*(_QWORD *)(a1 + 16), *(_DWORD *)(v9 + 112));
    sub_2F687C0(v9, v162, v118, v119, (__int64 *)v9);
    v84 = v157;
    v90 = *(_QWORD *)(a1 + 40);
LABEL_119:
    v155 = 0;
    v152 = 0;
    v144 = *(_QWORD *)(v90 + 32);
    if ( *(_QWORD *)(v9 + 104) )
    {
      v123 = v84;
      v126 = v9;
      v91 = *(_QWORD **)(v9 + 104);
      do
      {
        v120 = sub_2F658F0((__int64)v91, v26);
        if ( v120 )
        {
          v152 |= v91[14];
          v92 = *(_QWORD **)(a1 + 24);
          v155 |= v91[15];
          v170 = 0;
          v124 = v92;
          v93 = (_QWORD *)sub_22077B0(0x28u);
          if ( v93 )
          {
            v93[1] = v91;
            v93[2] = v24;
            *v93 = v162;
            v93[3] = v120;
            v93[4] = &v166;
          }
          v169.m128i_i64[0] = (__int64)v93;
          v171 = sub_2F683C0;
          v170 = sub_2F610C0;
          sub_2E0C490(v12, v162, v91[14], v91[15], (unsigned __int64)&v169, v144, v124, 0);
          if ( v170 )
            v170((unsigned __int64 *)&v169, (const __m128i **)&v169, 3);
        }
        v91 = (_QWORD *)v91[13];
      }
      while ( v91 );
      v9 = v126;
      v84 = v123;
    }
    if ( *(_QWORD *)(v12 + 104) )
    {
      v163 = v9;
      v94 = *(_QWORD *)(v12 + 104);
      v95 = v12;
      v96 = v84;
      do
      {
        *(_QWORD *)&v97 = sub_2F612A0(*(_QWORD *)(v94 + 112), *(_QWORD *)(v94 + 120), v152);
        if ( v97 == 0 )
        {
          v98 = (const __m128i *)sub_2E09D00((__int64 *)v94, v24);
          if ( v98 != (const __m128i *)(*(_QWORD *)v94 + 24LL * *(unsigned int *)(v94 + 8)) )
          {
            v99 = v98->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_DWORD *)(v99 + 24) | (unsigned int)(v98->m128i_i64[0] >> 1) & 3) <= (*(_DWORD *)(v23 + 24) | 2u)
              && v23 == v99 )
            {
              v169 = _mm_loadu_si128(v98);
              v170 = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v98[1].m128i_i64[0];
              sub_2E0C3B0(v94, v169.m128i_i64[0], v169.m128i_i64[1], 1);
            }
          }
        }
        v94 = *(_QWORD *)(v94 + 104);
      }
      while ( v94 );
      v9 = v163;
      v84 = v96;
      v12 = v95;
    }
  }
  v164 = v84;
  *(_QWORD *)(v158 + 8) = *(_QWORD *)(v84 + 8);
  v100 = sub_2F60D70(v12, v158, v9, v84, v85, v83);
  v166 |= BYTE1(v100);
  sub_2E14FC0(*(_QWORD *)(a1 + 40), v9, *(_QWORD *)(v164 + 8));
  LOBYTE(result) = 1;
  HIBYTE(result) = v166;
  return result;
}
