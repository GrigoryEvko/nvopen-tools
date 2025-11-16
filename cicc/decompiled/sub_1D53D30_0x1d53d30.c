// Function: sub_1D53D30
// Address: 0x1d53d30
//
__int64 __fastcall sub_1D53D30(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  _BYTE *v6; // rsi
  unsigned __int64 **v7; // rdi
  __int64 *v8; // rcx
  const __m128i *v9; // rdx
  unsigned __int64 v10; // r14
  __m128i *v11; // rax
  __m128i *v12; // rcx
  __m128i *v13; // rax
  __m128i *v14; // rax
  __int8 *v15; // rax
  const __m128i *v16; // rcx
  signed __int64 v17; // r14
  __m128i *v18; // rax
  __m128i *v19; // rcx
  __m128i *v20; // rax
  __m128i *v21; // rax
  __int8 *v22; // rax
  __int64 v23; // r15
  __int64 *v24; // rax
  __int64 v25; // r12
  __int64 *v26; // rbx
  __int64 v27; // rax
  bool v28; // zf
  __int64 *v29; // rax
  __int64 *v30; // rdx
  __int64 *v31; // rbx
  __int64 *v32; // r13
  char v33; // dl
  int v34; // r8d
  int v35; // r9d
  __int64 v36; // r10
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 v39; // r14
  __int64 *v40; // rsi
  __int64 *v41; // rcx
  __int64 v42; // rax
  unsigned __int64 *v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rax
  int v46; // edx
  unsigned __int64 *v47; // rcx
  unsigned __int64 *v48; // rax
  char v49; // r13
  __int64 v50; // rsi
  __int64 v51; // rbx
  __int64 (*v52)(void); // rax
  _QWORD *v53; // r14
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // r13
  __int64 *v57; // rbx
  __int64 v58; // r8
  __int64 v59; // r12
  __int64 v60; // rax
  int v61; // eax
  __int64 v62; // rsi
  int v63; // edx
  __int64 v64; // rax
  __int64 (__fastcall *v65)(__int64, unsigned __int8); // r13
  unsigned int v66; // eax
  __int64 v67; // rsi
  __int64 v68; // rsi
  __int64 v69; // rbx
  __int64 v70; // rax
  __int64 v71; // r13
  __int64 v72; // r8
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r14
  unsigned __int64 v77; // r15
  int v78; // edx
  __int64 v79; // r14
  __int64 v80; // [rsp+0h] [rbp-3F0h]
  __int64 v81; // [rsp+8h] [rbp-3E8h]
  __int64 v82; // [rsp+20h] [rbp-3D0h]
  __int64 v83; // [rsp+30h] [rbp-3C0h]
  __int64 *v84; // [rsp+58h] [rbp-398h]
  __int64 *v85; // [rsp+60h] [rbp-390h]
  __int64 v86; // [rsp+70h] [rbp-380h]
  __int64 *v87; // [rsp+70h] [rbp-380h]
  unsigned __int64 *v88; // [rsp+70h] [rbp-380h]
  __int64 *v89; // [rsp+78h] [rbp-378h]
  unsigned int v90; // [rsp+80h] [rbp-370h]
  char v91; // [rsp+86h] [rbp-36Ah]
  bool v92; // [rsp+87h] [rbp-369h]
  __int64 *v93; // [rsp+90h] [rbp-360h]
  _QWORD *v94; // [rsp+98h] [rbp-358h]
  __int64 v95; // [rsp+A0h] [rbp-350h] BYREF
  __int64 v96; // [rsp+A8h] [rbp-348h]
  __int64 v97; // [rsp+B0h] [rbp-340h]
  __int64 v98; // [rsp+C0h] [rbp-330h] BYREF
  _QWORD *v99; // [rsp+C8h] [rbp-328h]
  _QWORD *v100; // [rsp+D0h] [rbp-320h]
  __int64 v101; // [rsp+D8h] [rbp-318h]
  int v102; // [rsp+E0h] [rbp-310h]
  _QWORD v103[8]; // [rsp+E8h] [rbp-308h] BYREF
  const __m128i *v104; // [rsp+128h] [rbp-2C8h] BYREF
  const __m128i *v105; // [rsp+130h] [rbp-2C0h]
  __int64 v106; // [rsp+138h] [rbp-2B8h]
  __int64 *v107[16]; // [rsp+140h] [rbp-2B0h] BYREF
  __int64 *v108[2]; // [rsp+1C0h] [rbp-230h] BYREF
  __int64 *v109; // [rsp+1D0h] [rbp-220h]
  _BYTE v110[64]; // [rsp+1E8h] [rbp-208h] BYREF
  __m128i *v111; // [rsp+228h] [rbp-1C8h]
  __m128i *v112; // [rsp+230h] [rbp-1C0h]
  __int8 *v113; // [rsp+238h] [rbp-1B8h]
  __int64 v114; // [rsp+240h] [rbp-1B0h] BYREF
  __int64 v115; // [rsp+248h] [rbp-1A8h]
  unsigned __int64 v116; // [rsp+250h] [rbp-1A0h]
  __int64 v117; // [rsp+258h] [rbp-198h]
  __int64 v118; // [rsp+260h] [rbp-190h]
  char v119[64]; // [rsp+268h] [rbp-188h] BYREF
  __m128i *v120; // [rsp+2A8h] [rbp-148h]
  __m128i *v121; // [rsp+2B0h] [rbp-140h]
  __int8 *v122; // [rsp+2B8h] [rbp-138h]
  unsigned __int64 *v123; // [rsp+2C0h] [rbp-130h] BYREF
  __int64 v124; // [rsp+2C8h] [rbp-128h]
  unsigned __int64 v125[3]; // [rsp+2D0h] [rbp-120h] BYREF
  _BYTE v126[64]; // [rsp+2E8h] [rbp-108h] BYREF
  __m128i *v127; // [rsp+328h] [rbp-C8h]
  __m128i *v128; // [rsp+330h] [rbp-C0h]
  __int8 *v129; // [rsp+338h] [rbp-B8h]
  __m128i v130; // [rsp+340h] [rbp-B0h] BYREF
  __int64 *v131; // [rsp+350h] [rbp-A0h]
  __int64 v132; // [rsp+358h] [rbp-98h]
  int v133; // [rsp+360h] [rbp-90h]
  _BYTE v134[64]; // [rsp+368h] [rbp-88h] BYREF
  __m128i *v135; // [rsp+3A8h] [rbp-48h]
  __m128i *v136; // [rsp+3B0h] [rbp-40h]
  __int8 *v137; // [rsp+3B8h] [rbp-38h]

  v82 = *(_QWORD *)(a1 + 16);
  result = *(_QWORD *)(*(_QWORD *)v82 + 1160LL);
  if ( (__int64 (*)())result == sub_1D45FE0 )
    return result;
  result = ((__int64 (__fastcall *)(__int64))result)(v82);
  v91 = result;
  if ( !(_BYTE)result )
    return result;
  result = *(unsigned int *)(a1 + 192);
  if ( !(_DWORD)result )
    return result;
  v3 = *(_QWORD *)(a1 + 8);
  v95 = 0;
  v4 = *(_QWORD *)(v3 + 328);
  v104 = 0;
  memset(v107, 0, sizeof(v107));
  LODWORD(v107[3]) = 8;
  v107[1] = (__int64 *)&v107[5];
  v107[2] = (__int64 *)&v107[5];
  v99 = v103;
  v100 = v103;
  v103[0] = v4;
  v105 = 0;
  v106 = 0;
  v101 = 0x100000008LL;
  v102 = 0;
  v98 = 1;
  v5 = *(_QWORD *)(v4 + 88);
  v130.m128i_i64[0] = v4;
  v130.m128i_i64[1] = v5;
  v96 = 0;
  v97 = 0;
  sub_1D530F0(&v104, 0, &v130);
  sub_1D53270((__int64)&v98);
  v6 = v126;
  v7 = &v123;
  sub_16CCCB0(&v123, (__int64)v126, (__int64)v107);
  v8 = v107[14];
  v9 = (const __m128i *)v107[13];
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v10 = (char *)v107[14] - (char *)v107[13];
  if ( v107[14] == v107[13] )
  {
    v11 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_142;
    v11 = (__m128i *)sub_22077B0((char *)v107[14] - (char *)v107[13]);
    v8 = v107[14];
    v9 = (const __m128i *)v107[13];
  }
  v127 = v11;
  v128 = v11;
  v129 = &v11->m128i_i8[v10];
  if ( v8 == (__int64 *)v9 )
  {
    v12 = v11;
  }
  else
  {
    v12 = (__m128i *)((char *)v11 + (char *)v8 - (char *)v9);
    do
    {
      if ( v11 )
        *v11 = _mm_loadu_si128(v9);
      ++v11;
      ++v9;
    }
    while ( v11 != v12 );
  }
  v128 = v12;
  sub_16CCEE0(&v130, (__int64)v134, 8, (__int64)&v123);
  v13 = v127;
  v6 = v110;
  v127 = 0;
  v135 = v13;
  v14 = v128;
  v128 = 0;
  v136 = v14;
  v15 = v129;
  v129 = 0;
  v137 = v15;
  v7 = (unsigned __int64 **)v108;
  sub_16CCCB0(v108, (__int64)v110, (__int64)&v98);
  v16 = v105;
  v9 = v104;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  if ( v105 == v104 )
  {
    v17 = 0;
    v18 = 0;
    goto LABEL_17;
  }
  v17 = (char *)v105 - (char *)v104;
  if ( (unsigned __int64)((char *)v105 - (char *)v104) > 0x7FFFFFFFFFFFFFF0LL )
LABEL_142:
    sub_4261EA(v7, v6, v9);
  v18 = (__m128i *)sub_22077B0((char *)v105 - (char *)v104);
  v16 = v105;
  v9 = v104;
LABEL_17:
  v111 = v18;
  v112 = v18;
  v113 = &v18->m128i_i8[v17];
  if ( v9 == v16 )
  {
    v19 = v18;
  }
  else
  {
    v19 = (__m128i *)((char *)v18 + (char *)v16 - (char *)v9);
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v9);
      ++v18;
      ++v9;
    }
    while ( v18 != v19 );
  }
  v112 = v19;
  sub_16CCEE0(&v114, (__int64)v119, 8, (__int64)v108);
  v20 = v111;
  v111 = 0;
  v120 = v20;
  v21 = v112;
  v112 = 0;
  v121 = v21;
  v22 = v113;
  v113 = 0;
  v122 = v22;
  sub_1D533B0((__int64)&v114, (__int64)&v130, (__int64)&v95);
  if ( v120 )
    j_j___libc_free_0(v120, v122 - (__int8 *)v120);
  if ( v116 != v115 )
    _libc_free(v116);
  if ( v111 )
    j_j___libc_free_0(v111, v113 - (__int8 *)v111);
  if ( v109 != v108[1] )
    _libc_free((unsigned __int64)v109);
  if ( v135 )
    j_j___libc_free_0(v135, v137 - (__int8 *)v135);
  if ( v131 != (__int64 *)v130.m128i_i64[1] )
    _libc_free((unsigned __int64)v131);
  if ( v127 )
    j_j___libc_free_0(v127, v129 - (__int8 *)v127);
  if ( v125[0] != v124 )
    _libc_free(v125[0]);
  if ( v104 )
    j_j___libc_free_0(v104, v106 - (_QWORD)v104);
  if ( v100 != v99 )
    _libc_free((unsigned __int64)v100);
  if ( v107[13] )
    j_j___libc_free_0(v107[13], (char *)v107[15] - (char *)v107[13]);
  if ( v107[2] != v107[1] )
    _libc_free((unsigned __int64)v107[2]);
  result = v96;
  v83 = v96;
  v80 = v95;
  if ( v95 != v96 )
  {
    v23 = a1;
    while ( 1 )
    {
      v89 = (__int64 *)(v23 + 112);
      v84 = (__int64 *)(v23 + 80);
      v94 = *(_QWORD **)(v83 - 8);
      v24 = *(__int64 **)(v23 + 184);
      v93 = v24;
      v85 = &v24[*(unsigned int *)(v23 + 192)];
      if ( v24 == v85 )
        goto LABEL_93;
      do
      {
        v25 = *v93;
        v130.m128i_i64[0] = (__int64)v94;
        v130.m128i_i64[1] = v25;
        sub_1D4AC10(v107, v89, v130.m128i_i64);
        v130.m128i_i64[0] = (__int64)v94;
        v26 = v107[2];
        v130.m128i_i64[1] = v25;
        sub_1D4AC10(v108, v84, v130.m128i_i64);
        v27 = *(_QWORD *)(v23 + 120) + 24LL * *(unsigned int *)(v23 + 136);
        if ( (__int64 *)v27 == v26 )
        {
          if ( v109 != (__int64 *)(*(_QWORD *)(v23 + 88) + 24LL * *(unsigned int *)(v23 + 104)) )
            goto LABEL_92;
          v90 = 0;
        }
        else
        {
          v90 = *((_DWORD *)v26 + 4);
        }
        v28 = v27 == (_QWORD)v26;
        v130.m128i_i64[0] = 0;
        v132 = 8;
        v123 = v125;
        v124 = 0x400000000LL;
        v29 = (__int64 *)v134;
        v133 = 0;
        v130.m128i_i64[1] = (__int64)v134;
        v30 = (__int64 *)v134;
        v131 = (__int64 *)v134;
        v31 = (__int64 *)v94[9];
        v92 = !v28;
        if ( v31 == (__int64 *)v94[8] )
          goto LABEL_103;
        v32 = (__int64 *)v94[8];
        while ( 1 )
        {
          v39 = *v32;
          if ( v30 != v29 )
            goto LABEL_53;
          v40 = &v29[HIDWORD(v132)];
          if ( v40 == v29 )
            break;
          v41 = 0;
          while ( v39 != *v29 )
          {
            if ( *v29 == -2 )
              v41 = v29;
            if ( v40 == ++v29 )
            {
              if ( !v41 )
                goto LABEL_105;
              *v41 = v39;
              --v133;
              ++v130.m128i_i64[0];
              goto LABEL_54;
            }
          }
LABEL_57:
          if ( v31 == ++v32 )
            goto LABEL_70;
LABEL_58:
          v30 = v131;
          v29 = (__int64 *)v130.m128i_i64[1];
        }
LABEL_105:
        if ( HIDWORD(v132) >= (unsigned int)v132 )
        {
LABEL_53:
          sub_16CCBA0((__int64)&v130, *v32);
          if ( !v33 )
            goto LABEL_57;
        }
        else
        {
          ++HIDWORD(v132);
          *v40 = v39;
          ++v130.m128i_i64[0];
        }
LABEL_54:
        v36 = (unsigned int)sub_1FE54B0(v23, v39, v25);
        v37 = (unsigned int)v124;
        if ( (unsigned int)v124 >= HIDWORD(v124) )
        {
          v86 = v36;
          sub_16CD150((__int64)&v123, v125, 0, 16, v34, v35);
          v37 = (unsigned int)v124;
          v36 = v86;
        }
        v38 = (__int64 *)&v123[2 * v37];
        *v38 = v39;
        v38[1] = v36;
        LODWORD(v124) = v124 + 1;
        if ( v94 != (_QWORD *)v39 || v92 )
          goto LABEL_57;
        v115 = v25;
        ++v32;
        v114 = (__int64)v94;
        sub_1D4AC10(v107, v89, &v114);
        v90 = *((_DWORD *)v107[2] + 4);
        v92 = v91;
        if ( v31 != v32 )
          goto LABEL_58;
LABEL_70:
        if ( !(_DWORD)v124 )
          goto LABEL_103;
        v42 = 16LL * (unsigned int)v124;
        v43 = &v123[(unsigned __int64)v42 / 8];
        v44 = v42 >> 4;
        v45 = v42 >> 6;
        if ( !v45 )
        {
          v48 = v123;
LABEL_98:
          switch ( v44 )
          {
            case 2LL:
              v63 = *((_DWORD *)v123 + 2);
              break;
            case 3LL:
              v63 = *((_DWORD *)v123 + 2);
              if ( *((_DWORD *)v48 + 2) != v63 )
                goto LABEL_78;
              v48 += 2;
              break;
            case 1LL:
              v63 = *((_DWORD *)v123 + 2);
              goto LABEL_102;
            default:
              goto LABEL_103;
          }
          if ( v63 != *((_DWORD *)v48 + 2) )
            goto LABEL_78;
          v48 += 2;
LABEL_102:
          if ( v63 != *((_DWORD *)v48 + 2) )
            goto LABEL_78;
LABEL_103:
          v49 = 0;
          if ( v92 )
            goto LABEL_79;
          sub_1FE5190(v23, v94, v25, *((unsigned int *)v123 + 2));
          goto LABEL_87;
        }
        v46 = *((_DWORD *)v123 + 2);
        v47 = &v123[8 * v45];
        v48 = v123;
        while ( v46 == *((_DWORD *)v48 + 6) )
        {
          if ( v46 != *((_DWORD *)v48 + 10) )
          {
            v48 += 4;
            goto LABEL_78;
          }
          if ( v46 != *((_DWORD *)v48 + 14) )
          {
            v48 += 6;
            goto LABEL_78;
          }
          v48 += 8;
          if ( v47 == v48 )
          {
            v44 = ((char *)v43 - (char *)v48) >> 4;
            goto LABEL_98;
          }
          if ( *((_DWORD *)v48 + 2) != v46 )
            goto LABEL_78;
        }
        v48 += 2;
LABEL_78:
        v49 = v91;
        if ( v43 == v48 )
          goto LABEL_103;
LABEL_79:
        if ( *(_BYTE *)(v25 + 16) <= 0x17u )
        {
          v98 = 0;
        }
        else
        {
          v50 = *(_QWORD *)(v25 + 48);
          v98 = v50;
          if ( v50 )
            sub_1623A60((__int64)&v98, v50, 2);
        }
        v51 = 0;
        v52 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(v23 + 8) + 16LL) + 40LL);
        if ( v52 != sub_1D00B00 )
          v51 = v52();
        v53 = v94 + 2;
        if ( !v49 )
        {
          v54 = *(_QWORD *)(v51 + 8);
          v55 = sub_1DD5D10(v94);
          v56 = v94[7];
          v57 = (__int64 *)v55;
          v59 = sub_1E0B640(v56, v54 + 960, &v98, 0, v58);
          sub_1DD5BA0(v53, v59);
          v60 = *v57;
          *(_QWORD *)(v59 + 8) = v57;
          *(_QWORD *)v59 = v60 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v59 & 7LL;
          *(_QWORD *)((v60 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v59;
          *v57 = v59 | *v57 & 7;
          v114 = 0x10000000;
          v116 = 0;
          LODWORD(v115) = v90;
          v117 = 0;
          v118 = 0;
          sub_1E1A9C0(v59, v56, &v114);
          v61 = *((_DWORD *)v123 + 2);
          v114 = 0;
          v116 = 0;
          LODWORD(v115) = v61;
          v117 = 0;
          v118 = 0;
          sub_1E1A9C0(v59, v56, &v114);
          v62 = v98;
          if ( v98 )
            goto LABEL_86;
          goto LABEL_87;
        }
        v64 = sub_1E0A0C0(*(_QWORD *)(v23 + 8));
        v65 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v82 + 288LL);
        v66 = 8 * sub_15A9520(v64, 0);
        if ( v66 == 32 )
        {
          v67 = 5;
          goto LABEL_112;
        }
        if ( v66 > 0x20 )
        {
          v67 = 6;
          if ( v66 == 64 )
            goto LABEL_112;
          v67 = 0;
          if ( v66 == 128 )
            v67 = 7;
          if ( v65 == sub_1D45FB0 )
            goto LABEL_113;
LABEL_126:
          v68 = v65(v82, v67);
          if ( !v92 )
            goto LABEL_127;
          goto LABEL_114;
        }
        v67 = 3;
        if ( v66 != 8 )
        {
          LOBYTE(v67) = v66 == 16;
          v67 = (unsigned int)(4 * v67);
        }
LABEL_112:
        if ( v65 != sub_1D45FB0 )
          goto LABEL_126;
LABEL_113:
        v68 = *(_QWORD *)(v82 + 8 * (v67 & 7) + 120);
        if ( v92 )
          goto LABEL_114;
LABEL_127:
        v90 = sub_1E6B9A0(*(_QWORD *)(*(_QWORD *)(v23 + 8) + 40LL), v68, byte_3F871B3, 0);
LABEL_114:
        v69 = *(_QWORD *)(v51 + 8);
        v70 = sub_1DD5D10(v94);
        v71 = v94[7];
        v87 = (__int64 *)v70;
        v73 = sub_1E0B640(v71, v69, &v98, 0, v72);
        sub_1DD5BA0(v53, v73);
        v74 = *(_QWORD *)v73 & 7LL;
        v75 = *v87;
        *(_QWORD *)(v73 + 8) = v87;
        v75 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v73 = v75 | v74;
        *(_QWORD *)(v75 + 8) = v73;
        *v87 = v73 | *v87 & 7;
        v114 = 0x10000000;
        LODWORD(v115) = v90;
        v116 = 0;
        v117 = 0;
        v118 = 0;
        sub_1E1A9C0(v73, v71, &v114);
        v76 = 2LL * (unsigned int)v124;
        v88 = &v123[v76];
        if ( &v123[v76] != v123 )
        {
          v81 = v23;
          v77 = (unsigned __int64)v123;
          do
          {
            v78 = *(_DWORD *)(v77 + 8);
            v79 = *(_QWORD *)v77;
            v114 = 0;
            v77 += 16LL;
            LODWORD(v115) = v78;
            v116 = 0;
            v117 = 0;
            v118 = 0;
            sub_1E1A9C0(v73, v71, &v114);
            LOBYTE(v114) = 4;
            v116 = 0;
            LODWORD(v114) = v114 & 0xFFF000FF;
            v117 = v79;
            sub_1E1A9C0(v73, v71, &v114);
          }
          while ( v88 != (unsigned __int64 *)v77 );
          v23 = v81;
        }
        if ( !v92 )
          sub_1FE5190(v23, v94, v25, v90);
        v62 = v98;
        if ( v98 )
LABEL_86:
          sub_161E7C0((__int64)&v98, v62);
LABEL_87:
        if ( v131 != (__int64 *)v130.m128i_i64[1] )
          _libc_free((unsigned __int64)v131);
        if ( v123 != v125 )
          _libc_free((unsigned __int64)v123);
LABEL_92:
        ++v93;
      }
      while ( v85 != v93 );
LABEL_93:
      v83 -= 8;
      if ( v80 == v83 )
      {
        result = v95;
        v83 = v95;
        break;
      }
    }
  }
  if ( v83 )
    return j_j___libc_free_0(v83, v97 - v83);
  return result;
}
