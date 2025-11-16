// Function: sub_2144790
// Address: 0x2144790
//
void __fastcall sub_2144790(
        __int64 *a1,
        unsigned __int64 a2,
        __m128i *a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  unsigned __int8 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int128 v13; // xmm1
  __int64 v14; // r14
  __int64 v15; // rax
  bool v16; // cc
  unsigned __int64 v17; // rax
  __int64 v18; // r15
  __m128i v19; // xmm0
  unsigned int v20; // edx
  unsigned __int8 v21; // al
  __int64 v22; // rax
  __int64 v23; // r10
  unsigned int v24; // edx
  __int64 v25; // r15
  unsigned int v26; // eax
  __int64 *v27; // r10
  __int64 v28; // rcx
  unsigned __int64 v29; // r8
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // rcx
  unsigned __int64 v33; // r8
  __int64 *v34; // rax
  __int64 v35; // rsi
  __int32 v36; // edx
  unsigned int v37; // eax
  __int64 *v38; // r10
  __int64 v39; // rsi
  __int64 v40; // rcx
  unsigned __int64 v41; // r8
  __int64 v42; // rax
  unsigned __int64 v43; // rax
  __int64 v44; // r8
  unsigned int v45; // r14d
  __int64 v46; // rcx
  __int64 *v47; // rax
  __int64 v48; // rsi
  int v49; // edx
  const __m128i *v50; // r9
  __m128i v51; // xmm0
  _QWORD *v52; // rax
  __int64 v53; // rax
  unsigned int v54; // eax
  __int64 v55; // rdx
  __int64 v56; // rsi
  unsigned __int64 v57; // r15
  unsigned int v58; // esi
  int v59; // eax
  unsigned int v60; // esi
  int v61; // eax
  unsigned int v62; // eax
  __int64 v63; // rdx
  __int64 v64; // rsi
  unsigned __int64 v65; // r9
  __int64 v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  unsigned int v71; // esi
  int v72; // eax
  unsigned __int64 v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // rax
  unsigned int v76; // esi
  unsigned __int64 v77; // rax
  __int64 *v78; // [rsp+0h] [rbp-210h]
  __int64 *v79; // [rsp+8h] [rbp-208h]
  __int64 v80; // [rsp+8h] [rbp-208h]
  __int64 *v81; // [rsp+8h] [rbp-208h]
  __int64 v82; // [rsp+10h] [rbp-200h]
  __int64 v83; // [rsp+10h] [rbp-200h]
  __int64 *v84; // [rsp+10h] [rbp-200h]
  __int64 v85; // [rsp+10h] [rbp-200h]
  __int64 v86; // [rsp+18h] [rbp-1F8h]
  __int64 *v87; // [rsp+18h] [rbp-1F8h]
  __int64 v88; // [rsp+18h] [rbp-1F8h]
  __int64 *v89; // [rsp+18h] [rbp-1F8h]
  __int64 *v90; // [rsp+18h] [rbp-1F8h]
  __int64 v91; // [rsp+18h] [rbp-1F8h]
  __int64 v92; // [rsp+18h] [rbp-1F8h]
  __int64 v93; // [rsp+18h] [rbp-1F8h]
  int v94; // [rsp+20h] [rbp-1F0h]
  unsigned __int64 v95; // [rsp+20h] [rbp-1F0h]
  __int64 v96; // [rsp+20h] [rbp-1F0h]
  unsigned __int64 v97; // [rsp+20h] [rbp-1F0h]
  __int64 v98; // [rsp+20h] [rbp-1F0h]
  __int64 v99; // [rsp+20h] [rbp-1F0h]
  __int64 v100; // [rsp+20h] [rbp-1F0h]
  __int64 v101; // [rsp+20h] [rbp-1F0h]
  __int64 v102; // [rsp+20h] [rbp-1F0h]
  unsigned __int64 v103; // [rsp+20h] [rbp-1F0h]
  char v104; // [rsp+2Fh] [rbp-1E1h]
  __int64 *v105; // [rsp+30h] [rbp-1E0h]
  __int64 *v106; // [rsp+30h] [rbp-1E0h]
  __int64 v107; // [rsp+30h] [rbp-1E0h]
  __int64 v108; // [rsp+30h] [rbp-1E0h]
  __int64 v109; // [rsp+30h] [rbp-1E0h]
  __int64 v110; // [rsp+30h] [rbp-1E0h]
  __int64 v111; // [rsp+30h] [rbp-1E0h]
  __int64 v112; // [rsp+30h] [rbp-1E0h]
  unsigned __int64 v113; // [rsp+30h] [rbp-1E0h]
  __int64 v114; // [rsp+30h] [rbp-1E0h]
  __int64 v115; // [rsp+48h] [rbp-1C8h]
  unsigned __int64 v116; // [rsp+48h] [rbp-1C8h]
  __int64 v117; // [rsp+48h] [rbp-1C8h]
  unsigned __int64 v118; // [rsp+48h] [rbp-1C8h]
  __int64 v119; // [rsp+48h] [rbp-1C8h]
  __int64 v120; // [rsp+48h] [rbp-1C8h]
  __int64 v121; // [rsp+50h] [rbp-1C0h]
  __int64 v122; // [rsp+50h] [rbp-1C0h]
  __int64 *v123; // [rsp+50h] [rbp-1C0h]
  __int64 v124; // [rsp+50h] [rbp-1C0h]
  __int64 v125; // [rsp+50h] [rbp-1C0h]
  __int64 v126; // [rsp+50h] [rbp-1C0h]
  __int64 v127; // [rsp+68h] [rbp-1A8h]
  __int64 v129; // [rsp+88h] [rbp-188h]
  const __m128i *v130; // [rsp+88h] [rbp-188h]
  __int64 v131; // [rsp+88h] [rbp-188h]
  __m128i v132; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v133; // [rsp+D0h] [rbp-140h] BYREF
  int v134; // [rsp+D8h] [rbp-138h]
  __int64 v135; // [rsp+E0h] [rbp-130h] BYREF
  int v136; // [rsp+E8h] [rbp-128h]
  unsigned __int64 v137[2]; // [rsp+F0h] [rbp-120h] BYREF
  _OWORD v138[4]; // [rsp+100h] [rbp-110h] BYREF
  unsigned __int64 v139[2]; // [rsp+140h] [rbp-D0h] BYREF
  _OWORD v140[4]; // [rsp+150h] [rbp-C0h] BYREF
  __m128i *v141; // [rsp+190h] [rbp-80h] BYREF
  __int64 v142; // [rsp+198h] [rbp-78h]
  __m128i v143[7]; // [rsp+1A0h] [rbp-70h] BYREF

  v9 = *(unsigned __int8 **)(a2 + 40);
  v104 = *v9;
  sub_1F40D10((__int64)&v141, *a1, *(_QWORD *)(a1[1] + 48), *v9, *((_QWORD *)v9 + 1));
  v10 = *(_QWORD *)(a2 + 72);
  v132.m128i_i8[0] = v142;
  v132.m128i_i64[1] = v143[0].m128i_i64[0];
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(_QWORD *)v11;
  v13 = (__int128)_mm_loadu_si128((const __m128i *)(v11 + 40));
  v133 = v10;
  v129 = v12;
  v127 = *(_QWORD *)(v11 + 8);
  if ( v10 )
    sub_1623A60((__int64)&v133, v10, 2);
  v134 = *(_DWORD *)(a2 + 64);
  v14 = sub_1E0A0C0(*(_QWORD *)(a1[1] + 32));
  v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL) + 88LL);
  v16 = *(_DWORD *)(v15 + 32) <= 0x40u;
  v17 = *(_QWORD *)(v15 + 24);
  if ( !v16 )
    v17 = *(_QWORD *)v17;
  v18 = a1[1];
  v19 = _mm_load_si128(&v132);
  v139[0] = (unsigned __int64)v140;
  v138[0] = v19;
  v137[0] = (unsigned __int64)v138;
  v121 = (v17 | 8) & -(__int64)(v17 | 8);
  v137[1] = 0x400000001LL;
  v139[1] = 0x400000001LL;
  v140[0] = v19;
  v20 = 8 * sub_15A9520(v14, 0);
  if ( v20 == 32 )
  {
    v21 = 5;
  }
  else if ( v20 > 0x20 )
  {
    v21 = 6;
    if ( v20 != 64 )
    {
      v21 = 0;
      if ( v20 == 128 )
        v21 = 7;
    }
  }
  else
  {
    v21 = 3;
    if ( v20 != 8 )
      v21 = 4 * (v20 == 16);
  }
  v22 = sub_1D38BB0(v18, 0, (__int64)&v133, v21, 0, 1, v19, *(double *)&v13, a7, 0);
  v23 = a1[1];
  v143[0].m128i_i64[0] = v22;
  v143[0].m128i_i64[1] = v24;
  v141 = v143;
  v142 = 0x400000001LL;
  v105 = (__int64 *)v23;
  v25 = sub_1F58E60((__int64)&v132, *(_QWORD **)(v23 + 48));
  v26 = sub_15A9FE0(v14, v25);
  v27 = v105;
  v28 = 1;
  v29 = v26;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v25 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v53 = *(_QWORD *)(v25 + 32);
        v25 = *(_QWORD *)(v25 + 24);
        v28 *= v53;
        continue;
      case 1:
        v30 = 16;
        goto LABEL_16;
      case 2:
        v30 = 32;
        goto LABEL_16;
      case 3:
      case 9:
        v30 = 64;
        goto LABEL_16;
      case 4:
        v30 = 80;
        goto LABEL_16;
      case 5:
      case 6:
        v30 = 128;
        goto LABEL_16;
      case 7:
        v89 = v105;
        v58 = 0;
        v97 = v29;
        v111 = v28;
        goto LABEL_47;
      case 0xB:
        v30 = *(_DWORD *)(v25 + 8) >> 8;
        goto LABEL_16;
      case 0xD:
        v87 = v105;
        v95 = v29;
        v109 = v28;
        v52 = (_QWORD *)sub_15A9930(v14, v25);
        v28 = v109;
        v29 = v95;
        v27 = v87;
        v30 = 8LL * *v52;
        goto LABEL_16;
      case 0xE:
        v79 = v105;
        v82 = v29;
        v88 = v28;
        v96 = *(_QWORD *)(v25 + 24);
        v110 = *(_QWORD *)(v25 + 32);
        v54 = sub_15A9FE0(v14, v96);
        v27 = v79;
        v29 = v82;
        v55 = 1;
        v56 = v96;
        v28 = v88;
        v57 = v54;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v56 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v70 = *(_QWORD *)(v56 + 32);
              v56 = *(_QWORD *)(v56 + 24);
              v55 *= v70;
              continue;
            case 1:
              v68 = 16;
              goto LABEL_66;
            case 2:
              v68 = 32;
              goto LABEL_66;
            case 3:
            case 9:
              v68 = 64;
              goto LABEL_66;
            case 4:
              v68 = 80;
              goto LABEL_66;
            case 5:
            case 6:
              v68 = 128;
              goto LABEL_66;
            case 7:
              v71 = 0;
              v99 = v55;
              goto LABEL_76;
            case 0xB:
              v68 = *(_DWORD *)(v56 + 8) >> 8;
              goto LABEL_66;
            case 0xD:
              v101 = v55;
              v74 = (_QWORD *)sub_15A9930(v14, v56);
              v55 = v101;
              v28 = v88;
              v29 = v82;
              v27 = v79;
              v68 = 8LL * *v74;
              goto LABEL_66;
            case 0xE:
              v78 = v79;
              v80 = v82;
              v83 = v88;
              v91 = v55;
              v100 = *(_QWORD *)(v56 + 32);
              v73 = sub_12BE0A0(v14, *(_QWORD *)(v56 + 24));
              v55 = v91;
              v28 = v83;
              v29 = v80;
              v27 = v78;
              v68 = 8 * v100 * v73;
              goto LABEL_66;
            case 0xF:
              v71 = *(_DWORD *)(v56 + 8) >> 8;
              v99 = v55;
LABEL_76:
              v72 = sub_15A9520(v14, v71);
              v55 = v99;
              v28 = v88;
              v29 = v82;
              v27 = v79;
              v68 = (unsigned int)(8 * v72);
LABEL_66:
              v30 = 8 * v57 * v110 * ((v57 + ((unsigned __int64)(v68 * v55 + 7) >> 3) - 1) / v57);
              break;
          }
          goto LABEL_16;
        }
      case 0xF:
        v89 = v105;
        v97 = v29;
        v111 = v28;
        v58 = *(_DWORD *)(v25 + 8) >> 8;
LABEL_47:
        v59 = sub_15A9520(v14, v58);
        v28 = v111;
        v29 = v97;
        v27 = v89;
        v30 = (unsigned int)(8 * v59);
LABEL_16:
        v135 = v133;
        v31 = v29 + ((unsigned __int64)(v30 * v28 + 7) >> 3) - 1;
        v32 = *(_QWORD *)(a2 + 32);
        v33 = v31 / v29 * v29;
        if ( v133 )
        {
          v86 = *(_QWORD *)(a2 + 32);
          v94 = v33;
          v106 = v27;
          sub_1623A60((__int64)&v135, v133, 2);
          v32 = v86;
          LODWORD(v33) = v94;
          v27 = v106;
        }
        v136 = v134;
        v34 = sub_1D38FD0(
                v27,
                (__int64)v137,
                (__int64)&v135,
                v129,
                v127,
                v121,
                v19,
                v13,
                *(_OWORD *)(v32 + 80),
                v33,
                v143);
        v35 = v135;
        a3->m128i_i64[0] = (__int64)v34;
        a3->m128i_i32[2] = v36;
        if ( v35 )
          sub_161E7C0((__int64)&v135, v35);
        v107 = a1[1];
        v130 = v141;
        v122 = sub_1F58E60((__int64)&v132, *(_QWORD **)(v107 + 48));
        v37 = sub_15A9FE0(v14, v122);
        v38 = (__int64 *)v107;
        v39 = v122;
        v40 = 1;
        v41 = v37;
        while ( 1 )
        {
          switch ( *(_BYTE *)(v39 + 8) )
          {
            case 1:
              v42 = 16;
              goto LABEL_24;
            case 2:
              v42 = 32;
              goto LABEL_24;
            case 3:
            case 9:
              v42 = 64;
              goto LABEL_24;
            case 4:
              v42 = 80;
              goto LABEL_24;
            case 5:
            case 6:
              v42 = 128;
              goto LABEL_24;
            case 7:
              v60 = 0;
              v116 = v41;
              v124 = v40;
              goto LABEL_53;
            case 0xB:
              v42 = *(_DWORD *)(v39 + 8) >> 8;
              goto LABEL_24;
            case 0xD:
              v118 = v41;
              v126 = v40;
              v67 = (_QWORD *)sub_15A9930(v14, v39);
              v40 = v126;
              v41 = v118;
              v38 = (__int64 *)v107;
              v42 = 8LL * *v67;
              goto LABEL_24;
            case 0xE:
              v90 = (__int64 *)v107;
              v98 = v41;
              v112 = v40;
              v117 = *(_QWORD *)(v39 + 24);
              v125 = *(_QWORD *)(v39 + 32);
              v62 = sub_15A9FE0(v14, v117);
              v38 = v90;
              v41 = v98;
              v63 = 1;
              v64 = v117;
              v40 = v112;
              v65 = v62;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v64 + 8) )
                {
                  case 1:
                    v69 = 16;
                    goto LABEL_69;
                  case 2:
                    v69 = 32;
                    goto LABEL_69;
                  case 3:
                  case 9:
                    v69 = 64;
                    goto LABEL_69;
                  case 4:
                    v69 = 80;
                    goto LABEL_69;
                  case 5:
                  case 6:
                    v69 = 128;
                    goto LABEL_69;
                  case 7:
                    v84 = v90;
                    v76 = 0;
                    v92 = v98;
                    v102 = v112;
                    v113 = v65;
                    v119 = v63;
                    goto LABEL_85;
                  case 0xB:
                    v69 = *(_DWORD *)(v64 + 8) >> 8;
                    goto LABEL_69;
                  case 0xD:
                    v84 = v90;
                    v92 = v98;
                    v102 = v112;
                    v113 = v65;
                    v119 = v63;
                    v69 = 8LL * *(_QWORD *)sub_15A9930(v14, v64);
                    goto LABEL_86;
                  case 0xE:
                    v81 = v90;
                    v85 = v98;
                    v93 = v112;
                    v103 = v65;
                    v114 = v63;
                    v120 = *(_QWORD *)(v64 + 32);
                    v77 = sub_12BE0A0(v14, *(_QWORD *)(v64 + 24));
                    v63 = v114;
                    v65 = v103;
                    v40 = v93;
                    v41 = v85;
                    v38 = v81;
                    v69 = 8 * v120 * v77;
                    goto LABEL_69;
                  case 0xF:
                    v84 = v90;
                    v92 = v98;
                    v102 = v112;
                    v76 = *(_DWORD *)(v64 + 8) >> 8;
                    v113 = v65;
                    v119 = v63;
LABEL_85:
                    v69 = 8 * (unsigned int)sub_15A9520(v14, v76);
LABEL_86:
                    v63 = v119;
                    v65 = v113;
                    v40 = v102;
                    v41 = v92;
                    v38 = v84;
LABEL_69:
                    v42 = 8 * v125 * v65 * ((v65 + ((unsigned __int64)(v69 * v63 + 7) >> 3) - 1) / v65);
                    goto LABEL_24;
                  case 0x10:
                    v75 = *(_QWORD *)(v64 + 32);
                    v64 = *(_QWORD *)(v64 + 24);
                    v63 *= v75;
                    continue;
                  default:
                    goto LABEL_64;
                }
              }
            case 0xF:
              v116 = v41;
              v124 = v40;
              v60 = *(_DWORD *)(v39 + 8) >> 8;
LABEL_53:
              v61 = sub_15A9520(v14, v60);
              v40 = v124;
              v41 = v116;
              v38 = (__int64 *)v107;
              v42 = (unsigned int)(8 * v61);
LABEL_24:
              v135 = v133;
              v43 = v41 * ((v41 + ((unsigned __int64)(v42 * v40 + 7) >> 3) - 1) / v41);
              v44 = *(_QWORD *)(a2 + 32);
              v45 = v43;
              v46 = a3->m128i_i64[0];
              if ( v133 )
              {
                v108 = a3->m128i_i64[0];
                v115 = *(_QWORD *)(a2 + 32);
                v123 = v38;
                sub_1623A60((__int64)&v135, v133, 2);
                v46 = v108;
                v44 = v115;
                v38 = v123;
              }
              v136 = v134;
              v47 = sub_1D38FD0(
                      v38,
                      (__int64)v139,
                      (__int64)&v135,
                      v46,
                      1,
                      0,
                      v19,
                      v13,
                      *(_OWORD *)(v44 + 80),
                      v45,
                      v130);
              v48 = v135;
              *(_QWORD *)a4 = v47;
              *(_DWORD *)(a4 + 8) = v49;
              if ( v48 )
                sub_161E7C0((__int64)&v135, v48);
              v131 = *(_QWORD *)a4;
              if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1[1] + 32)) == 1 || v104 == 13 )
              {
                v51 = _mm_loadu_si128(a3);
                a3->m128i_i64[0] = *(_QWORD *)a4;
                a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
                *(_QWORD *)a4 = v51.m128i_i64[0];
                *(_DWORD *)(a4 + 8) = v51.m128i_i32[2];
              }
              sub_2013400((__int64)a1, a2, 1, v131, (__m128i *)(v127 & 0xFFFFFFFF00000000LL | 1), v50);
              if ( v141 != v143 )
                _libc_free((unsigned __int64)v141);
              if ( (_OWORD *)v139[0] != v140 )
                _libc_free(v139[0]);
              if ( (_OWORD *)v137[0] != v138 )
                _libc_free(v137[0]);
              if ( v133 )
                sub_161E7C0((__int64)&v133, v133);
              return;
            case 0x10:
              v66 = *(_QWORD *)(v39 + 32);
              v39 = *(_QWORD *)(v39 + 24);
              v40 *= v66;
              break;
            default:
LABEL_64:
              *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
              BUG();
          }
        }
    }
  }
}
