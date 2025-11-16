// Function: sub_2F67750
// Address: 0x2f67750
//
__int64 __fastcall sub_2F67750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v7; // cl
  __int64 v8; // r9
  int v9; // r14d
  unsigned __int64 v10; // rax
  unsigned int v11; // edx
  __int64 v12; // r15
  int v13; // r10d
  unsigned int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // r9
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 i; // rsi
  __int16 v21; // dx
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 *v25; // rdx
  __int64 v26; // r11
  __int64 v27; // rbx
  __int64 *v28; // rdx
  unsigned int v29; // r12d
  __int64 *v31; // r11
  unsigned int v32; // eax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 *v35; // rdx
  _DWORD *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  unsigned __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 *v41; // rcx
  __int64 v42; // r8
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned __int64 v45; // rsi
  int v46; // eax
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // r9
  __int64 *v52; // r11
  __int64 v53; // rdx
  __int64 v54; // r15
  __int64 v55; // r12
  unsigned __int64 v56; // rsi
  __int64 v57; // rdx
  const __m128i *v58; // rdx
  unsigned __int64 v59; // rsi
  __int64 v60; // r14
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  unsigned int v66; // edx
  __int64 v67; // rcx
  __int64 *v68; // rbx
  __int64 v69; // rax
  unsigned int v70; // edx
  __int64 v71; // rcx
  __int64 *v72; // rbx
  __int64 v73; // rax
  int v74; // edx
  __int64 v75; // r8
  unsigned __int64 v76; // r14
  __int64 *v77; // rdx
  __int64 *v78; // rsi
  __int64 v79; // r15
  unsigned __int64 v80; // r10
  _QWORD *v81; // rdx
  _QWORD *v82; // rsi
  int v83; // r8d
  unsigned int v84; // eax
  __int64 v85; // rax
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // r13
  __int64 *v90; // rdx
  __int64 v91; // r14
  __int64 v92; // rax
  __int128 v93; // [rsp-20h] [rbp-D0h]
  __int128 v94; // [rsp-20h] [rbp-D0h]
  __int64 *v95; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v96; // [rsp+10h] [rbp-A0h]
  __int64 *v97; // [rsp+18h] [rbp-98h]
  __int64 v98; // [rsp+20h] [rbp-90h]
  __int64 v99; // [rsp+28h] [rbp-88h]
  __int64 *v100; // [rsp+30h] [rbp-80h]
  __int64 *v101; // [rsp+30h] [rbp-80h]
  __int64 *v102; // [rsp+30h] [rbp-80h]
  __int64 v103; // [rsp+30h] [rbp-80h]
  unsigned __int64 v104; // [rsp+38h] [rbp-78h]
  __int64 *v105; // [rsp+38h] [rbp-78h]
  __int64 v106; // [rsp+38h] [rbp-78h]
  __int64 v107; // [rsp+38h] [rbp-78h]
  __int64 *v108; // [rsp+40h] [rbp-70h]
  __int64 v109; // [rsp+40h] [rbp-70h]
  __int64 v110; // [rsp+40h] [rbp-70h]
  __int64 v111; // [rsp+40h] [rbp-70h]
  __int64 *v112; // [rsp+40h] [rbp-70h]
  __int64 v113; // [rsp+48h] [rbp-68h]
  __int64 *v114; // [rsp+48h] [rbp-68h]
  __int64 v115; // [rsp+48h] [rbp-68h]
  __int64 v116; // [rsp+48h] [rbp-68h]
  unsigned __int64 v117; // [rsp+48h] [rbp-68h]
  int v118; // [rsp+48h] [rbp-68h]
  __int64 v119; // [rsp+48h] [rbp-68h]
  unsigned __int64 v120; // [rsp+50h] [rbp-60h]
  __int64 v121; // [rsp+50h] [rbp-60h]
  __int64 v122; // [rsp+50h] [rbp-60h]
  _QWORD *v123; // [rsp+50h] [rbp-60h]
  _QWORD *v124; // [rsp+50h] [rbp-60h]
  __int64 v125; // [rsp+50h] [rbp-60h]
  __int64 v126; // [rsp+50h] [rbp-60h]
  __m128i v128; // [rsp+60h] [rbp-50h]
  __int64 v129; // [rsp+70h] [rbp-40h]

  v7 = *(_BYTE *)(a2 + 26);
  v8 = *(_QWORD *)(a1 + 40);
  if ( v7 )
    v9 = *(_DWORD *)(a2 + 8);
  else
    v9 = *(_DWORD *)(a2 + 12);
  v10 = *(unsigned int *)(v8 + 160);
  v11 = v9 & 0x7FFFFFFF;
  if ( (v9 & 0x7FFFFFFFu) >= (unsigned int)v10 || (v12 = *(_QWORD *)(*(_QWORD *)(v8 + 152) + 8LL * v11)) == 0 )
  {
    v70 = v11 + 1;
    if ( (unsigned int)v10 < v70 && v70 != v10 )
    {
      if ( v70 >= v10 )
      {
        v79 = *(_QWORD *)(v8 + 168);
        v80 = v70 - v10;
        if ( v70 > (unsigned __int64)*(unsigned int *)(v8 + 164) )
        {
          v117 = v70 - v10;
          v125 = *(_QWORD *)(a1 + 40);
          sub_C8D5F0(v8 + 152, (const void *)(v8 + 168), v70, 8u, a5, v8);
          v8 = v125;
          v80 = v117;
          v10 = *(unsigned int *)(v125 + 160);
        }
        v71 = *(_QWORD *)(v8 + 152);
        v81 = (_QWORD *)(v71 + 8 * v10);
        v82 = &v81[v80];
        if ( v81 != v82 )
        {
          do
            *v81++ = v79;
          while ( v82 != v81 );
          LODWORD(v10) = *(_DWORD *)(v8 + 160);
          v71 = *(_QWORD *)(v8 + 152);
        }
        *(_DWORD *)(v8 + 160) = v80 + v10;
LABEL_59:
        v72 = (__int64 *)(v71 + 8LL * (v9 & 0x7FFFFFFF));
        v124 = (_QWORD *)v8;
        v73 = sub_2E10F30(v9);
        *v72 = v73;
        v12 = v73;
        sub_2E11E80(v124, v73);
        v8 = *(_QWORD *)(a1 + 40);
        v10 = *(unsigned int *)(v8 + 160);
        if ( *(_BYTE *)(a2 + 26) )
          goto LABEL_6;
LABEL_60:
        v13 = *(_DWORD *)(a2 + 8);
        goto LABEL_7;
      }
      *(_DWORD *)(v8 + 160) = v70;
    }
    v71 = *(_QWORD *)(v8 + 152);
    goto LABEL_59;
  }
  if ( !v7 )
    goto LABEL_60;
LABEL_6:
  v13 = *(_DWORD *)(a2 + 12);
LABEL_7:
  v14 = v13 & 0x7FFFFFFF;
  v15 = 8LL * (v13 & 0x7FFFFFFF);
  if ( (v13 & 0x7FFFFFFFu) >= (unsigned int)v10 || (v16 = *(_QWORD *)(*(_QWORD *)(v8 + 152) + 8LL * v14)) == 0 )
  {
    v66 = v14 + 1;
    if ( v66 > (unsigned int)v10 && v66 != v10 )
    {
      if ( v66 >= v10 )
      {
        v75 = *(_QWORD *)(v8 + 168);
        v76 = v66 - v10;
        if ( v66 > (unsigned __int64)*(unsigned int *)(v8 + 164) )
        {
          v111 = *(_QWORD *)(v8 + 168);
          v118 = v13;
          v126 = v8;
          sub_C8D5F0(v8 + 152, (const void *)(v8 + 168), v66, 8u, v75, v8);
          v8 = v126;
          v75 = v111;
          v13 = v118;
          v10 = *(unsigned int *)(v126 + 160);
        }
        v67 = *(_QWORD *)(v8 + 152);
        v77 = (__int64 *)(v67 + 8 * v10);
        v78 = &v77[v76];
        if ( v77 != v78 )
        {
          do
            *v77++ = v75;
          while ( v78 != v77 );
          LODWORD(v10) = *(_DWORD *)(v8 + 160);
          v67 = *(_QWORD *)(v8 + 152);
        }
        *(_DWORD *)(v8 + 160) = v76 + v10;
        goto LABEL_56;
      }
      *(_DWORD *)(v8 + 160) = v66;
    }
    v67 = *(_QWORD *)(v8 + 152);
LABEL_56:
    v68 = (__int64 *)(v67 + v15);
    v123 = (_QWORD *)v8;
    v69 = sub_2E10F30(v13);
    *v68 = v69;
    v16 = v69;
    sub_2E11E80(v123, v69);
    v8 = *(_QWORD *)(a1 + 40);
  }
  v17 = *(_QWORD *)(v8 + 32);
  v18 = a3;
  v19 = a3;
  if ( (*(_DWORD *)(a3 + 44) & 4) != 0 )
  {
    do
      v19 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v19 + 44) & 4) != 0 );
  }
  if ( (*(_DWORD *)(a3 + 44) & 8) != 0 )
  {
    do
      v18 = *(_QWORD *)(v18 + 8);
    while ( (*(_BYTE *)(v18 + 44) & 8) != 0 );
  }
  for ( i = *(_QWORD *)(v18 + 8); i != v19; v19 = *(_QWORD *)(v19 + 8) )
  {
    v21 = *(_WORD *)(v19 + 68);
    if ( (unsigned __int16)(v21 - 14) > 4u && v21 != 24 )
      break;
  }
  v22 = *(unsigned int *)(v17 + 144);
  v23 = *(_QWORD *)(v17 + 128);
  if ( (_DWORD)v22 )
  {
    v24 = (v22 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v25 = (__int64 *)(v23 + 16LL * v24);
    v26 = *v25;
    if ( v19 == *v25 )
      goto LABEL_19;
    v74 = 1;
    while ( v26 != -4096 )
    {
      v83 = v74 + 1;
      v24 = (v22 - 1) & (v74 + v24);
      v25 = (__int64 *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( *v25 == v19 )
        goto LABEL_19;
      v74 = v83;
    }
  }
  v25 = (__int64 *)(v23 + 16 * v22);
LABEL_19:
  v120 = v25[1] & 0xFFFFFFFFFFFFFFF8LL;
  v27 = v120 | 4;
  v28 = (__int64 *)sub_2E09D00((__int64 *)v16, v120 | 4);
  if ( v28 == (__int64 *)(*(_QWORD *)v16 + 24LL * *(unsigned int *)(v16 + 8)) )
    return 0;
  if ( (*(_DWORD *)((*v28 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v28 >> 1) & 3) > (*(_DWORD *)(v120 + 24) | 2u) )
    return 0;
  if ( *(_QWORD *)(v28[2] + 8) != v27 )
    return 0;
  v108 = v28;
  v113 = v28[2];
  v99 = v120 | 2;
  v31 = (__int64 *)sub_2E09D00((__int64 *)v12, v120 | 2);
  if ( v31 == (__int64 *)(*(_QWORD *)v12 + 24LL * *(unsigned int *)(v12 + 8)) )
    return 0;
  if ( (*(_DWORD *)((*v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v31 >> 1) & 3) > (*(_DWORD *)(v120 + 24) | 1u) )
    return 0;
  v100 = v108;
  v104 = v120;
  v98 = v31[2];
  v109 = v113;
  v114 = v31;
  v121 = *(_QWORD *)((*(_QWORD *)(v98 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
  LOBYTE(v32) = sub_2F66340((_DWORD *)a2, v121);
  v33 = v109;
  v34 = v104;
  v29 = v32;
  v35 = v100;
  if ( !(_BYTE)v32 )
    return 0;
  if ( *(_WORD *)(v121 + 68) != 20 )
    return 0;
  v36 = *(_DWORD **)(v121 + 32);
  if ( (*v36 & 0xFFF00) != 0 || (v36[10] & 0xFFF00) != 0 )
    return 0;
  v37 = *(_QWORD *)(v98 + 8);
  v38 = v37 >> 1;
  v39 = v37 & 0xFFFFFFFFFFFFFFF8LL;
  v40 = (v38 & 3) != 0 ? v39 | (2LL * (int)((v38 & 3) - 1)) : *(_QWORD *)v39 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v101 = v114;
  v105 = v35;
  v110 = v34;
  v115 = v33;
  v41 = (__int64 *)sub_2E09D00((__int64 *)v16, v40);
  if ( v41 == (__int64 *)(*(_QWORD *)v16 + 24LL * *(unsigned int *)(v16 + 8)) )
    return 0;
  v42 = v115;
  v43 = (__int64)v105;
  if ( (*(_DWORD *)((*v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v41 >> 1) & 3) > (*(_DWORD *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)(v40 >> 1) & 3) )
    return 0;
  v44 = v41[1];
  v45 = v44 & 0xFFFFFFFFFFFFFFF8LL;
  v46 = (v44 >> 1) & 3;
  v47 = v46 ? (v45 | (2LL * (v46 - 1))) & 0xFFFFFFFFFFFFFFF8LL : *(_QWORD *)v45 & 0xFFFFFFFFFFFFFFF8LL;
  v48 = *(_QWORD *)(v47 + 16);
  v122 = v48;
  if ( v48 && *(_QWORD *)(a3 + 24) == *(_QWORD *)(v48 + 24) && v105 == v41 + 3 )
  {
    v49 = *v105;
    v50 = v41[1];
    v129 = v115;
    v95 = v101;
    *(_QWORD *)(v115 + 8) = v50;
    *((_QWORD *)&v93 + 1) = v49;
    *(_QWORD *)&v93 = v50;
    v102 = v41;
    v106 = v115;
    v116 = v50;
    sub_2E0F080(v16, v45, v43, (__int64)v41, v42, v110, v93, v129);
    v51 = v110;
    v52 = v95;
    v53 = v102[2];
    if ( v53 != v106 )
    {
      sub_2E0AAF0(v16, v106, v53);
      v52 = v95;
      v51 = v110;
    }
    if ( *(_QWORD *)(v16 + 104) )
    {
      v107 = v12;
      v54 = v51;
      v96 = v29;
      v55 = *(_QWORD *)(v16 + 104);
      v103 = v16;
      v97 = v52;
      do
      {
        v58 = (const __m128i *)sub_2E09D00((__int64 *)v55, v27);
        if ( v58 != (const __m128i *)(*(_QWORD *)v55 + 24LL * *(unsigned int *)(v55 + 8))
          && (v59 = v58->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL,
              (*(_DWORD *)(v59 + 24) | (unsigned int)(v58->m128i_i64[0] >> 1) & 3) <= (*(_DWORD *)(v54 + 24) | 2u))
          && v59 == (v58->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v128 = _mm_loadu_si128(v58);
          sub_2E0C3B0(v55, v128.m128i_i64[0], v128.m128i_i64[1], 1);
        }
        else
        {
          if ( !sub_2F658F0(v55, v116) )
          {
            v91 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
            v92 = sub_2F65810(v91, v116);
            sub_2E0EAB0(v55, *(_QWORD *)(*(_QWORD *)(v91 + 152) + 16LL * *(unsigned int *)(v92 + 24)), v116);
          }
          v60 = sub_2F658F0(v55, v27);
          *((_QWORD *)&v94 + 1) = v49;
          *(_QWORD *)&v94 = v116;
          sub_2E0F080(v55, v27, v61, v62, v63, v64, v94, v60);
          v65 = *(_QWORD *)(v98 + 8);
          if ( ((v65 >> 1) & 3) != 0 )
            v56 = v65 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v65 >> 1) & 3) - 1));
          else
            v56 = *(_QWORD *)(v65 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
          v57 = sub_2F658F0(v55, v56);
          if ( v60 != v57 )
            sub_2E0AAF0(v55, v60, v57);
        }
        v55 = *(_QWORD *)(v55 + 104);
      }
      while ( v55 );
      v51 = v54;
      v16 = v103;
      v12 = v107;
      v52 = v97;
      v29 = v96;
    }
    v112 = v52;
    v119 = v51;
    v84 = sub_2E89C70(v122, *(_DWORD *)(v16 + 112), 0, 1);
    if ( v84 != -1 )
    {
      v85 = *(_QWORD *)(v122 + 32) + 40LL * v84;
      *(_BYTE *)(v85 + 3) &= ~0x40u;
    }
    sub_2E8A790(a3, *(_DWORD *)(v12 + 112), *(_DWORD *)(v16 + 112), 0, *(_QWORD **)(a1 + 24));
    v88 = v119;
    if ( v27 == v112[1] )
    {
LABEL_97:
      sub_2F60EC0(a1, v12, 0, v86, v87, v88);
      return v29;
    }
    v89 = *(_QWORD *)(v12 + 104);
    if ( v89 )
    {
      while ( 1 )
      {
        v90 = (__int64 *)sub_2E09D00((__int64 *)v89, v99);
        if ( v90 != (__int64 *)(*(_QWORD *)v89 + 24LL * *(unsigned int *)(v89 + 8)) )
        {
          v86 = *(_DWORD *)(v119 + 24) | 1u;
          if ( (*(_DWORD *)((*v90 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v90 >> 1) & 3) <= (unsigned int)v86
            && v27 == v90[1] )
          {
            break;
          }
        }
        v89 = *(_QWORD *)(v89 + 104);
        if ( !v89 )
          return v29;
      }
      goto LABEL_97;
    }
  }
  else
  {
    return 0;
  }
  return v29;
}
