// Function: sub_2090780
// Address: 0x2090780
//
void __fastcall sub_2090780(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r15
  unsigned int v9; // esi
  __int64 v10; // r14
  __int64 v11; // r9
  unsigned int v12; // edi
  __int64 v13; // rcx
  unsigned int v14; // r11d
  _QWORD *v15; // rax
  __int64 v16; // r10
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // r10
  _QWORD *v20; // rax
  __int64 v21; // r11
  __int64 v22; // rbx
  unsigned __int8 v23; // al
  int v24; // ecx
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r13
  _QWORD *v35; // r14
  __int64 v36; // rdx
  __int64 v37; // r15
  __int64 v38; // rcx
  __int64 v39; // r8
  int v40; // r9d
  __int64 *v41; // rcx
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rsi
  int v46; // edx
  __int64 *v47; // rbx
  int v48; // r14d
  _QWORD *v49; // r8
  int v50; // eax
  int v51; // ecx
  int v52; // esi
  int v53; // esi
  __int64 v54; // rdi
  int v55; // eax
  unsigned int v56; // ecx
  _QWORD *v57; // r8
  int v58; // r10d
  _QWORD *v59; // r11
  int v60; // eax
  __int64 v61; // rax
  unsigned int v62; // r13d
  unsigned int v63; // eax
  __int64 v64; // r13
  __int64 v65; // rcx
  __int64 v66; // rdi
  unsigned __int64 v67; // rax
  unsigned int v68; // r13d
  __int64 v69; // r14
  int v70; // ebx
  __int64 v71; // rax
  _QWORD *v72; // r12
  __int64 v73; // r15
  unsigned __int64 *v74; // rcx
  unsigned __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rbx
  __int64 v78; // r12
  __int64 v79; // r13
  __int64 *v80; // rdi
  int v81; // eax
  int v82; // esi
  __int64 v83; // rdx
  unsigned int v84; // eax
  __int64 v85; // rdi
  int v86; // r10d
  _QWORD *v87; // r9
  int v88; // eax
  int v89; // eax
  __int64 v90; // rdi
  int v91; // r10d
  unsigned int v92; // edx
  __int64 v93; // rsi
  int v94; // esi
  int v95; // esi
  __int64 v96; // r9
  _QWORD *v97; // r10
  int v98; // r11d
  unsigned int v99; // ecx
  __int64 v100; // rdi
  unsigned int v101; // r15d
  __int64 v102; // r13
  int v103; // r12d
  __int64 v104; // rdx
  __int64 v105; // rbx
  __int128 v106; // [rsp-10h] [rbp-F0h]
  unsigned int v107; // [rsp-10h] [rbp-F0h]
  __int64 v108; // [rsp+8h] [rbp-D8h]
  __int64 *v109; // [rsp+8h] [rbp-D8h]
  unsigned int v110; // [rsp+10h] [rbp-D0h]
  int v111; // [rsp+10h] [rbp-D0h]
  __int64 v112; // [rsp+10h] [rbp-D0h]
  __int64 v113; // [rsp+10h] [rbp-D0h]
  __int64 v114; // [rsp+10h] [rbp-D0h]
  unsigned int v115; // [rsp+18h] [rbp-C8h]
  int v116; // [rsp+18h] [rbp-C8h]
  __int64 v117; // [rsp+18h] [rbp-C8h]
  int v118; // [rsp+18h] [rbp-C8h]
  int v119; // [rsp+18h] [rbp-C8h]
  __int64 v120; // [rsp+18h] [rbp-C8h]
  __int64 v121; // [rsp+18h] [rbp-C8h]
  __int64 v122; // [rsp+20h] [rbp-C0h]
  __int64 *v123; // [rsp+20h] [rbp-C0h]
  unsigned int v124; // [rsp+20h] [rbp-C0h]
  __int64 v125; // [rsp+20h] [rbp-C0h]
  __int64 v126; // [rsp+50h] [rbp-90h] BYREF
  int v127; // [rsp+58h] [rbp-88h]
  __int64 v128; // [rsp+60h] [rbp-80h] BYREF
  __int64 v129; // [rsp+68h] [rbp-78h]
  __int64 v130; // [rsp+70h] [rbp-70h]
  __int64 v131; // [rsp+78h] [rbp-68h]
  __int64 v132; // [rsp+80h] [rbp-60h]
  __int64 v133; // [rsp+88h] [rbp-58h]
  __int64 v134; // [rsp+90h] [rbp-50h]
  __int64 v135; // [rsp+98h] [rbp-48h] BYREF
  int v136; // [rsp+A0h] [rbp-40h]
  __int64 v137; // [rsp+A8h] [rbp-38h]

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 712);
  v8 = *(_QWORD *)(a2 - 24);
  v9 = *(_DWORD *)(v7 + 72);
  v10 = *(_QWORD *)(v7 + 784);
  v11 = v7 + 48;
  if ( v9 )
  {
    v12 = v9 - 1;
    v13 = *(_QWORD *)(v7 + 56);
    v14 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v15 = (_QWORD *)(v13 + 16LL * v14);
    v16 = *v15;
    if ( v8 == *v15 )
    {
LABEL_3:
      v122 = v15[1];
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
      {
        v17 = *(_QWORD *)(a2 - 72);
        v18 = *(_QWORD *)(a2 - 48);
        goto LABEL_5;
      }
LABEL_24:
      sub_1DD8FE0(v10, v122, -1);
      if ( v122 != sub_2054600(v6, v10) || !(unsigned int)sub_1700720(*(_QWORD *)(v6 + 544)) )
      {
        v34 = *(_QWORD *)(v6 + 552);
        v35 = sub_1D2A490((_QWORD *)v34, v122, v30, v31, v32, v33);
        v37 = v36;
        v128 = 0;
        v41 = sub_2051DF0((__int64 *)v6, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v122, v36, v38, v39, v40);
        v43 = v42;
        v44 = *(_QWORD *)v6;
        LODWORD(v129) = *(_DWORD *)(v6 + 536);
        if ( v44 )
        {
          if ( &v128 != (__int64 *)(v44 + 48) )
          {
            v45 = *(_QWORD *)(v44 + 48);
            v128 = v45;
            if ( v45 )
            {
              v123 = v41;
              sub_1623A60((__int64)&v128, v45, 2);
              v41 = v123;
            }
          }
        }
        *((_QWORD *)&v106 + 1) = v37;
        *(_QWORD *)&v106 = v35;
        v47 = sub_1D332F0(
                (__int64 *)v34,
                188,
                (__int64)&v128,
                1,
                0,
                0,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                a5,
                (__int64)v41,
                v43,
                v106);
        v48 = v46;
        if ( v47 )
        {
          nullsub_686();
          *(_QWORD *)(v34 + 176) = v47;
          *(_DWORD *)(v34 + 184) = v48;
          sub_1D23870();
        }
        else
        {
          *(_QWORD *)(v34 + 176) = 0;
          *(_DWORD *)(v34 + 184) = v46;
        }
        if ( v128 )
          sub_161E7C0((__int64)&v128, v128);
      }
      return;
    }
    v116 = 1;
    v49 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v49 )
        v49 = v15;
      v14 = v12 & (v116 + v14);
      v15 = (_QWORD *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( v8 == *v15 )
        goto LABEL_3;
      ++v116;
    }
    if ( !v49 )
      v49 = v15;
    v50 = *(_DWORD *)(v7 + 64);
    ++*(_QWORD *)(v7 + 48);
    v51 = v50 + 1;
    if ( 4 * (v50 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(v7 + 68) - v51 <= v9 >> 3 )
      {
        v124 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
        sub_1D52F30(v7 + 48, v9);
        v88 = *(_DWORD *)(v7 + 72);
        if ( !v88 )
          goto LABEL_134;
        v89 = v88 - 1;
        v90 = *(_QWORD *)(v7 + 56);
        v87 = 0;
        v91 = 1;
        v92 = v89 & v124;
        v51 = *(_DWORD *)(v7 + 64) + 1;
        v49 = (_QWORD *)(v90 + 16LL * (v89 & v124));
        v93 = *v49;
        if ( v8 != *v49 )
        {
          while ( v93 != -8 )
          {
            if ( v93 == -16 && !v87 )
              v87 = v49;
            v92 = v89 & (v91 + v92);
            v49 = (_QWORD *)(v90 + 16LL * v92);
            v93 = *v49;
            if ( v8 == *v49 )
              goto LABEL_39;
            ++v91;
          }
LABEL_79:
          if ( v87 )
            v49 = v87;
          goto LABEL_39;
        }
      }
      goto LABEL_39;
    }
  }
  else
  {
    ++*(_QWORD *)(v7 + 48);
  }
  sub_1D52F30(v7 + 48, 2 * v9);
  v81 = *(_DWORD *)(v7 + 72);
  if ( !v81 )
    goto LABEL_134;
  v82 = v81 - 1;
  v83 = *(_QWORD *)(v7 + 56);
  v84 = (v81 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v51 = *(_DWORD *)(v7 + 64) + 1;
  v49 = (_QWORD *)(v83 + 16LL * v84);
  v85 = *v49;
  if ( v8 != *v49 )
  {
    v86 = 1;
    v87 = 0;
    while ( v85 != -8 )
    {
      if ( !v87 && v85 == -16 )
        v87 = v49;
      v84 = v82 & (v86 + v84);
      v49 = (_QWORD *)(v83 + 16LL * v84);
      v85 = *v49;
      if ( v8 == *v49 )
        goto LABEL_39;
      ++v86;
    }
    goto LABEL_79;
  }
LABEL_39:
  *(_DWORD *)(v7 + 64) = v51;
  if ( *v49 != -8 )
    --*(_DWORD *)(v7 + 68);
  *v49 = v8;
  v49[1] = 0;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 1 )
  {
    v122 = 0;
    goto LABEL_24;
  }
  v7 = *(_QWORD *)(v6 + 712);
  v17 = *(_QWORD *)(a2 - 72);
  v18 = *(_QWORD *)(a2 - 48);
  v9 = *(_DWORD *)(v7 + 72);
  v11 = v7 + 48;
  if ( !v9 )
  {
    ++*(_QWORD *)(v7 + 48);
    v117 = v18;
    v122 = 0;
    goto LABEL_44;
  }
  v122 = 0;
  v13 = *(_QWORD *)(v7 + 56);
  v12 = v9 - 1;
LABEL_5:
  v115 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
  v19 = v12 & v115;
  v20 = (_QWORD *)(v13 + 16 * v19);
  v21 = *v20;
  if ( v18 != *v20 )
  {
    v111 = 1;
    v57 = 0;
    while ( v21 != -8 )
    {
      if ( v21 != -16 || v57 )
        v20 = v57;
      LODWORD(v19) = v12 & (v111 + v19);
      v109 = (__int64 *)(v13 + 16LL * (unsigned int)v19);
      v21 = *v109;
      if ( v18 == *v109 )
      {
        v22 = v109[1];
        goto LABEL_7;
      }
      ++v111;
      v57 = v20;
      v20 = (_QWORD *)(v13 + 16LL * (unsigned int)v19);
    }
    if ( !v57 )
      v57 = v20;
    v60 = *(_DWORD *)(v7 + 64);
    ++*(_QWORD *)(v7 + 48);
    v55 = v60 + 1;
    if ( 4 * v55 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(v7 + 68) - v55 > v9 >> 3 )
      {
LABEL_57:
        *(_DWORD *)(v7 + 64) = v55;
        if ( *v57 != -8 )
          --*(_DWORD *)(v7 + 68);
        *v57 = v18;
        v22 = 0;
        v57[1] = 0;
        goto LABEL_7;
      }
      v114 = v18;
      sub_1D52F30(v11, v9);
      v94 = *(_DWORD *)(v7 + 72);
      if ( v94 )
      {
        v95 = v94 - 1;
        v96 = *(_QWORD *)(v7 + 56);
        v97 = 0;
        v18 = v114;
        v98 = 1;
        v99 = v95 & v115;
        v55 = *(_DWORD *)(v7 + 64) + 1;
        v57 = (_QWORD *)(v96 + 16LL * (v95 & v115));
        v100 = *v57;
        if ( v114 != *v57 )
        {
          while ( v100 != -8 )
          {
            if ( !v97 && v100 == -16 )
              v97 = v57;
            v99 = v95 & (v98 + v99);
            v57 = (_QWORD *)(v96 + 16LL * v99);
            v100 = *v57;
            if ( v114 == *v57 )
              goto LABEL_57;
            ++v98;
          }
          if ( v97 )
            v57 = v97;
        }
        goto LABEL_57;
      }
LABEL_134:
      ++*(_DWORD *)(v7 + 64);
      BUG();
    }
    v117 = v18;
LABEL_44:
    v110 = v18;
    sub_1D52F30(v11, 2 * v9);
    v52 = *(_DWORD *)(v7 + 72);
    if ( v52 )
    {
      v53 = v52 - 1;
      v54 = *(_QWORD *)(v7 + 56);
      v55 = *(_DWORD *)(v7 + 64) + 1;
      v56 = v53 & ((v110 >> 9) ^ (v110 >> 4));
      v57 = (_QWORD *)(v54 + 16LL * v56);
      v18 = *v57;
      if ( v117 != *v57 )
      {
        v58 = 1;
        v59 = 0;
        while ( v18 != -8 )
        {
          if ( !v59 && v18 == -16 )
            v59 = v57;
          v56 = v53 & (v58 + v56);
          v57 = (_QWORD *)(v54 + 16LL * v56);
          v18 = *v57;
          if ( v117 == *v57 )
            goto LABEL_57;
          ++v58;
        }
        v18 = v117;
        if ( v59 )
          v57 = v59;
      }
      goto LABEL_57;
    }
    goto LABEL_134;
  }
  v22 = v20[1];
LABEL_7:
  v23 = *(_BYTE *)(v17 + 16);
  if ( v23 <= 0x17u
    || (v24 = v23, (unsigned int)v23 - 35 > 0x11)
    || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 552) + 16LL) + 56LL)
    || (v25 = *(_QWORD *)(v17 + 8)) == 0
    || *(_QWORD *)(v25 + 8)
    || (*(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0) && (v118 = v24, v61 = sub_1625790(a2, 15), v24 = v118, v61)
    || (unsigned int)(v24 - 50) > 1 )
  {
LABEL_12:
    v26 = *(_DWORD *)(v6 + 536);
    v27 = *(_QWORD *)v6;
    v126 = 0;
    v127 = v26;
    if ( v27 )
    {
      if ( &v126 != (__int64 *)(v27 + 48) )
      {
        v28 = *(_QWORD *)(v27 + 48);
        v126 = v28;
        if ( v28 )
          sub_1623A60((__int64)&v126, v28, 2);
      }
    }
    v29 = sub_159C4F0(*(__int64 **)(*(_QWORD *)(v6 + 552) + 48LL));
    LODWORD(v128) = 17;
    v131 = v29;
    v129 = v17;
    v130 = 0;
    v132 = v122;
    v133 = v22;
    v134 = v10;
    v135 = v126;
    if ( v126 )
    {
      sub_1623A60((__int64)&v135, v126, 2);
      v137 = -1;
      v136 = v127;
      if ( v126 )
        sub_161E7C0((__int64)&v126, v126);
    }
    else
    {
      v137 = -1;
      v136 = v127;
    }
    sub_2069F40((__int64 *)v6, (__int64)&v128, v10, a3, a4, a5);
    if ( v135 )
      sub_161E7C0((__int64)&v135, v135);
    return;
  }
  v119 = v24;
  v62 = sub_2052E90(v6, v10, v22);
  v63 = sub_2052E90(v6, v10, v122);
  v107 = v62;
  v64 = v6 + 584;
  sub_2056210(v6, v17, v122, v22, v10, v10, v119 - 24, v63, v107, 0);
  if ( !sub_2052F70(v6, v6 + 584) )
  {
    v66 = *(_QWORD *)(v6 + 592);
    v120 = *(_QWORD *)(v6 + 584);
    v65 = v120;
    v67 = 0xCCCCCCCCCCCCCCCDLL * ((v66 - v120) >> 4);
    if ( (_DWORD)v67 != 1 )
    {
      v121 = v10;
      v68 = 1;
      v69 = v6;
      v108 = v22;
      v70 = v67;
      v112 = v17;
      while ( 1 )
      {
        v71 = v68++;
        v72 = *(_QWORD **)(v65 + 80 * v71 + 48);
        v73 = *(_QWORD *)(*(_QWORD *)(v69 + 712) + 8LL) + 320LL;
        sub_1DD5B80(v73, (__int64)v72);
        v74 = (unsigned __int64 *)v72[1];
        v75 = *v72 & 0xFFFFFFFFFFFFFFF8LL;
        *v74 = v75 | *v74 & 7;
        *(_QWORD *)(v75 + 8) = v74;
        *v72 &= 7uLL;
        v72[1] = 0;
        sub_1E0A230(v73, v72);
        if ( v70 == v68 )
          break;
        v65 = *(_QWORD *)(v69 + 584);
      }
      v6 = v69;
      v10 = v121;
      v17 = v112;
      v22 = v108;
      v66 = *(_QWORD *)(v6 + 592);
      v120 = *(_QWORD *)(v6 + 584);
    }
    if ( v120 != v66 )
    {
      v76 = v6;
      v113 = v22;
      v77 = v66;
      v78 = v120;
      v79 = v76;
      do
      {
        v80 = (__int64 *)(v78 + 56);
        v78 += 80;
        sub_17CD270(v80);
      }
      while ( v77 != v78 );
      v22 = v113;
      v6 = v79;
      *(_QWORD *)(v79 + 592) = v120;
    }
    goto LABEL_12;
  }
  v101 = 1;
  if ( -858993459 * (unsigned int)((__int64)(*(_QWORD *)(v6 + 592) - *(_QWORD *)(v6 + 584)) >> 4) != 1 )
  {
    v125 = v6 + 584;
    v102 = v6;
    v103 = -858993459 * ((__int64)(*(_QWORD *)(v6 + 592) - *(_QWORD *)(v6 + 584)) >> 4);
    do
    {
      v104 = v101++;
      v105 = 80 * v104;
      sub_2090460(v102, *(_QWORD *)(*(_QWORD *)(v102 + 584) + 80 * v104 + 8), a3, a4, a5);
      sub_2090460(v102, *(_QWORD *)(*(_QWORD *)(v102 + 584) + v105 + 24), a3, a4, a5);
    }
    while ( v103 != v101 );
    v6 = v102;
    v64 = v125;
  }
  sub_2069F40((__int64 *)v6, *(_QWORD *)(v6 + 584), v10, a3, a4, a5);
  sub_20566E0(v64, *(_QWORD *)(v6 + 584));
}
