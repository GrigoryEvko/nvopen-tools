// Function: sub_37594D0
// Address: 0x37594d0
//
void __fastcall sub_37594D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rsi
  int v11; // edx
  __int64 v12; // rdi
  unsigned int i; // eax
  __int64 v14; // rcx
  __int64 v15; // r14
  int v16; // eax
  int v17; // edx
  int v18; // ebx
  char v19; // dl
  __int64 v20; // rsi
  int v21; // eax
  unsigned int v22; // ecx
  const char *v23; // rcx
  unsigned int v24; // r15d
  __int64 v25; // rcx
  int v26; // eax
  unsigned int v27; // edx
  __int64 v28; // rcx
  int v29; // eax
  unsigned int v30; // edx
  __int64 v31; // rcx
  int v32; // eax
  unsigned int v33; // edx
  __int64 v34; // rcx
  int v35; // eax
  unsigned int v36; // edx
  __int64 v37; // rcx
  int v38; // eax
  unsigned int v39; // edx
  __int64 v40; // rcx
  int v41; // eax
  unsigned int v42; // edx
  __int64 v43; // rcx
  int v44; // eax
  unsigned int v45; // edx
  __int64 v46; // rcx
  int v47; // eax
  unsigned int v48; // edx
  __int64 v49; // rcx
  int v50; // eax
  unsigned int v51; // edx
  int v52; // eax
  __int64 v53; // rax
  __int16 v54; // r9
  __int64 v55; // rdi
  __m128i *v56; // rax
  __m128i si128; // xmm0
  int v58; // eax
  int v59; // r14d
  int v60; // r13d
  int v61; // r12d
  int v62; // ebx
  int v63; // r15d
  __int64 v64; // rdi
  int v65; // r9d
  __int64 v66; // rdi
  __int64 v67; // rdi
  _BYTE *v68; // rax
  __int64 v69; // rax
  int v70; // eax
  int v71; // eax
  int v72; // eax
  int v73; // eax
  int v74; // eax
  int v75; // eax
  int v76; // eax
  int v77; // eax
  int v78; // eax
  __int64 v79; // rax
  int v80; // r8d
  int v81; // r10d
  unsigned int v82; // r9d
  const char *v83; // rax
  int v84; // r11d
  int v85; // edi
  int v86; // r8d
  unsigned int v87; // r9d
  int v88; // r10d
  int v89; // eax
  int v90; // eax
  __int64 v91; // rdi
  __int64 v92; // rax
  int v93; // eax
  int v94; // edx
  unsigned int v95; // eax
  const char *v96; // rcx
  __int64 v97; // rdi
  __int64 v98; // rax
  __m128i *v99; // rax
  __m128i v100; // xmm0
  __int64 v101; // rax
  __m128i *v102; // rax
  __m128i v103; // xmm0
  __int64 v104; // r8
  unsigned int v105; // edi
  __int64 v106; // rax
  int v107; // eax
  int v108; // edx
  int v109; // r8d
  int v110; // eax
  __int64 v111; // rax
  __m128i *v112; // rax
  __m128i v113; // xmm0
  int v114; // r15d
  int v115; // r11d
  __int64 v116; // rax
  unsigned __int64 v117; // rdx
  int v118; // ecx
  __int64 v119; // rax
  __int64 v120; // rdi
  void *v121; // rax
  __m128i *v122; // rax
  __m128i v123; // xmm0
  __int64 v124; // rax
  void *v125; // rax
  __int64 v126; // rax
  void *v127; // rax
  __int64 v128; // rax
  __m128i *v129; // rax
  __m128i v130; // xmm0
  __int64 v131; // rax
  void *v132; // rax
  __int64 v133; // rax
  __m128i *v134; // rax
  __m128i v135; // xmm0
  __int64 v136; // rax
  __int64 v137; // rdi
  void *v138; // rax
  __int64 v139; // rax
  void *v140; // rax
  __int64 v141; // rax
  __m128i *v142; // rax
  __m128i v143; // xmm0
  const char *v144; // [rsp+18h] [rbp-118h]
  __int64 v145; // [rsp+20h] [rbp-110h]
  __int64 v146; // [rsp+28h] [rbp-108h]
  __int64 v147; // [rsp+30h] [rbp-100h]
  int v148; // [rsp+3Ch] [rbp-F4h]
  int v149; // [rsp+3Ch] [rbp-F4h]
  int v150; // [rsp+3Ch] [rbp-F4h]
  int v151; // [rsp+3Ch] [rbp-F4h]
  int v152; // [rsp+3Ch] [rbp-F4h]
  int v153; // [rsp+3Ch] [rbp-F4h]
  unsigned __int64 v154; // [rsp+40h] [rbp-F0h]
  int v155; // [rsp+40h] [rbp-F0h]
  __int64 v156; // [rsp+48h] [rbp-E8h]
  unsigned int v157; // [rsp+48h] [rbp-E8h]
  __int16 v158; // [rsp+48h] [rbp-E8h]
  int v159; // [rsp+48h] [rbp-E8h]
  int v160; // [rsp+48h] [rbp-E8h]
  int v161[8]; // [rsp+50h] [rbp-E0h] BYREF
  _BYTE *v162; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v163; // [rsp+78h] [rbp-B8h]
  _BYTE v164[176]; // [rsp+80h] [rbp-B0h] BYREF

  v162 = v164;
  v163 = 0x1000000000LL;
  v8 = *(_QWORD *)(a1 + 8);
  v145 = v8 + 400;
  v147 = *(_QWORD *)(v8 + 408);
  if ( v147 != v8 + 400 )
  {
    v146 = v6;
    v144 = (const char *)(a1 + 520);
    while ( 1 )
    {
      if ( !v147 )
        goto LABEL_237;
      v154 = v147 - 8;
      if ( *(_DWORD *)(v147 + 28) == -1 )
      {
        v116 = (unsigned int)v163;
        v117 = (unsigned int)v163 + 1LL;
        if ( v117 > HIDWORD(v163) )
        {
          sub_C8D5F0((__int64)&v162, v164, v117, 8u, a5, a6);
          v116 = (unsigned int)v163;
        }
        *(_QWORD *)&v162[8 * v116] = v154;
        LODWORD(v163) = v163 + 1;
      }
      if ( !*(_DWORD *)(v147 + 60) )
        goto LABEL_18;
      v156 = *(unsigned int *)(v147 + 60);
      v9 = 0;
      do
      {
        if ( (*(_BYTE *)(a1 + 304) & 1) != 0 )
        {
          v10 = a1 + 312;
          v11 = 7;
        }
        else
        {
          v17 = *(_DWORD *)(a1 + 320);
          v10 = *(_QWORD *)(a1 + 312);
          if ( !v17 )
            goto LABEL_16;
          v11 = v17 - 1;
        }
        v12 = (unsigned int)v9;
        a5 = 1;
        for ( i = v11 & (v9 + ((v154 >> 9) ^ (v154 >> 4))); ; i = v11 & v16 )
        {
          v14 = v10 + 24LL * i;
          v15 = *(_QWORD *)v14;
          if ( v154 == *(_QWORD *)v14 )
            break;
          if ( !v15 && *(_DWORD *)(v14 + 8) == -1 )
            goto LABEL_16;
LABEL_12:
          v16 = a5 + i;
          a5 = (unsigned int)(a5 + 1);
        }
        if ( *(_DWORD *)(v14 + 8) != (_DWORD)v9 )
          goto LABEL_12;
        v18 = *(_DWORD *)(v14 + 16);
        if ( !v18 )
        {
LABEL_16:
          if ( *(_DWORD *)(v147 + 28) != -3 )
            goto LABEL_17;
          HIWORD(v105) = WORD1(v146);
          v106 = *(_QWORD *)(v147 + 40) + 16 * v9;
          LOWORD(v105) = *(_WORD *)v106;
          sub_2FE6CC0(
            (__int64)v161,
            *(_QWORD *)a1,
            *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL),
            v105,
            *(_QWORD *)(v106 + 8));
          if ( !LOBYTE(v161[0]) )
            goto LABEL_17;
          v107 = *(_DWORD *)(v147 + 16);
          if ( v107 == 9 || v107 == 35 )
            goto LABEL_17;
          v18 = 0;
          v24 = 0;
          if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
            goto LABEL_126;
          goto LABEL_142;
        }
        v19 = *(_BYTE *)(a1 + 1536) & 1;
        if ( v19 )
        {
          v20 = a1 + 1544;
          v21 = 7;
          goto LABEL_26;
        }
        v69 = *(unsigned int *)(a1 + 1552);
        v20 = *(_QWORD *)(a1 + 1544);
        if ( !(_DWORD)v69 )
          goto LABEL_136;
        v21 = v69 - 1;
LABEL_26:
        v22 = v21 & (37 * v18);
        a5 = v20 + 8LL * v22;
        v12 = *(unsigned int *)a5;
        if ( v18 != (_DWORD)v12 )
        {
          v109 = 1;
          while ( (_DWORD)v12 != -1 )
          {
            a6 = (unsigned int)(v109 + 1);
            v22 = v21 & (v109 + v22);
            a5 = v20 + 8LL * v22;
            v12 = *(unsigned int *)a5;
            if ( v18 == (_DWORD)v12 )
              goto LABEL_27;
            v109 = a6;
          }
          if ( v19 )
          {
            v104 = 64;
          }
          else
          {
            v69 = *(unsigned int *)(a1 + 1552);
LABEL_136:
            v104 = 8 * v69;
          }
          a5 = v20 + v104;
        }
LABEL_27:
        if ( !v19 )
        {
          v79 = *(unsigned int *)(a1 + 1552);
          v23 = (const char *)(v20 + 8 * v79);
          v12 = v79;
          if ( (const char *)a5 == v23 )
          {
LABEL_29:
            v24 = 0;
            goto LABEL_30;
          }
          v80 = *(_DWORD *)(a5 + 4);
          v81 = v79 - 1;
          v161[0] = v80;
          if ( (_DWORD)v79 )
            goto LABEL_106;
LABEL_186:
          v83 = v23;
          goto LABEL_107;
        }
        v23 = (const char *)(v20 + 64);
        if ( a5 == v20 + 64 )
          goto LABEL_29;
        v80 = *(_DWORD *)(a5 + 4);
        v81 = 7;
        LODWORD(v12) = 8;
        v161[0] = v80;
LABEL_106:
        v82 = v81 & (37 * v80);
        v83 = (const char *)(v20 + 8LL * v82);
        v84 = *(_DWORD *)v83;
        if ( *(_DWORD *)v83 != v80 )
        {
          v110 = 1;
          while ( v84 != -1 )
          {
            v114 = v110 + 1;
            v82 = v81 & (v110 + v82);
            v83 = (const char *)(v20 + 8LL * v82);
            v84 = *(_DWORD *)v83;
            if ( *(_DWORD *)v83 == v80 )
              goto LABEL_107;
            v110 = v114;
          }
          goto LABEL_186;
        }
LABEL_107:
        v85 = v12 - 1;
LABEL_108:
        while ( v19 )
        {
LABEL_109:
          if ( v83 == v23 )
            goto LABEL_118;
          v86 = *((_DWORD *)v83 + 1);
          v161[0] = v86;
LABEL_111:
          v87 = v85 & (37 * v86);
          v83 = (const char *)(v20 + 8LL * v87);
          v88 = *(_DWORD *)v83;
          if ( *(_DWORD *)v83 != v86 )
          {
            v89 = 1;
            while ( v88 != -1 )
            {
              v115 = v89 + 1;
              v87 = v85 & (v89 + v87);
              v83 = (const char *)(v20 + 8LL * v87);
              v88 = *(_DWORD *)v83;
              if ( *(_DWORD *)v83 == v86 )
                goto LABEL_108;
              v89 = v115;
            }
            goto LABEL_114;
          }
        }
        while ( v83 != v23 )
        {
          v86 = *((_DWORD *)v83 + 1);
          v90 = *(_DWORD *)(a1 + 1552);
          v161[0] = v86;
          if ( v90 )
            goto LABEL_111;
LABEL_114:
          v83 = v23;
          if ( v19 )
            goto LABEL_109;
        }
LABEL_118:
        v20 = (__int64)v161;
        v12 = a1;
        v24 = 1;
        sub_37593F0(a1, v161);
LABEL_30:
        if ( (*(_BYTE *)(a1 + 720) & 1) != 0 )
        {
          v25 = a1 + 728;
          v26 = 7;
LABEL_32:
          v27 = v26 & (37 * v18);
          v20 = *(unsigned int *)(v25 + 8LL * v27);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_33:
            v24 |= 2u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v27 = v26 & (v12 + v27);
              v20 = *(unsigned int *)(v25 + 8LL * v27);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_33;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v78 = *(_DWORD *)(a1 + 736);
          v25 = *(_QWORD *)(a1 + 728);
          if ( v78 )
          {
            v26 = v78 - 1;
            goto LABEL_32;
          }
        }
        if ( (*(_BYTE *)(a1 + 912) & 1) != 0 )
        {
          v28 = a1 + 920;
          v29 = 7;
LABEL_36:
          v30 = v29 & (37 * v18);
          v20 = *(unsigned int *)(v28 + 8LL * v30);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_37:
            v24 |= 4u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v30 = v29 & (v12 + v30);
              v20 = *(unsigned int *)(v28 + 8LL * v30);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_37;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v77 = *(_DWORD *)(a1 + 928);
          v28 = *(_QWORD *)(a1 + 920);
          if ( v77 )
          {
            v29 = v77 - 1;
            goto LABEL_36;
          }
        }
        if ( (*(_BYTE *)(a1 + 1264) & 1) != 0 )
        {
          v31 = a1 + 1272;
          v32 = 7;
LABEL_40:
          v33 = v32 & (37 * v18);
          v20 = *(unsigned int *)(v31 + 8LL * v33);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_41:
            v24 |= 8u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v33 = v32 & (v12 + v33);
              v20 = *(unsigned int *)(v31 + 8LL * v33);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_41;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v76 = *(_DWORD *)(a1 + 1280);
          v31 = *(_QWORD *)(a1 + 1272);
          if ( v76 )
          {
            v32 = v76 - 1;
            goto LABEL_40;
          }
        }
        if ( (*(_BYTE *)(a1 + 800) & 1) != 0 )
        {
          v34 = a1 + 808;
          v35 = 7;
LABEL_44:
          v36 = v35 & (37 * v18);
          v20 = *(unsigned int *)(v34 + 12LL * v36);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_45:
            v24 |= 0x10u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v36 = v35 & (v12 + v36);
              v20 = *(unsigned int *)(v34 + 12LL * v36);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_45;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v75 = *(_DWORD *)(a1 + 816);
          v34 = *(_QWORD *)(a1 + 808);
          if ( v75 )
          {
            v35 = v75 - 1;
            goto LABEL_44;
          }
        }
        if ( (*(_BYTE *)(a1 + 1152) & 1) != 0 )
        {
          v37 = a1 + 1160;
          v38 = 7;
LABEL_48:
          v39 = v38 & (37 * v18);
          v20 = *(unsigned int *)(v37 + 12LL * v39);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_49:
            v24 |= 0x20u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v39 = v38 & (v12 + v39);
              v20 = *(unsigned int *)(v37 + 12LL * v39);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_49;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v74 = *(_DWORD *)(a1 + 1168);
          v37 = *(_QWORD *)(a1 + 1160);
          if ( v74 )
          {
            v38 = v74 - 1;
            goto LABEL_48;
          }
        }
        if ( (*(_BYTE *)(a1 + 1344) & 1) != 0 )
        {
          v40 = a1 + 1352;
          v41 = 7;
LABEL_52:
          v42 = v41 & (37 * v18);
          v20 = *(unsigned int *)(v40 + 12LL * v42);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_53:
            v24 |= 0x40u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v42 = v41 & (v12 + v42);
              v20 = *(unsigned int *)(v40 + 12LL * v42);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_53;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v73 = *(_DWORD *)(a1 + 1360);
          v40 = *(_QWORD *)(a1 + 1352);
          if ( v73 )
          {
            v41 = v73 - 1;
            goto LABEL_52;
          }
        }
        if ( (*(_BYTE *)(a1 + 1456) & 1) != 0 )
        {
          v43 = a1 + 1464;
          v44 = 7;
LABEL_56:
          v45 = v44 & (37 * v18);
          v20 = *(unsigned int *)(v43 + 8LL * v45);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_57:
            LOBYTE(v24) = v24 | 0x80;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v45 = v44 & (v12 + v45);
              v20 = *(unsigned int *)(v43 + 8LL * v45);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_57;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v72 = *(_DWORD *)(a1 + 1472);
          v43 = *(_QWORD *)(a1 + 1464);
          if ( v72 )
          {
            v44 = v72 - 1;
            goto LABEL_56;
          }
        }
        if ( (*(_BYTE *)(a1 + 992) & 1) != 0 )
        {
          v46 = a1 + 1000;
          v47 = 7;
LABEL_60:
          v48 = v47 & (37 * v18);
          v20 = *(unsigned int *)(v46 + 8LL * v48);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_61:
            v24 |= 0x100u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v48 = v47 & (v12 + v48);
              v20 = *(unsigned int *)(v46 + 8LL * v48);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_61;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v71 = *(_DWORD *)(a1 + 1008);
          v46 = *(_QWORD *)(a1 + 1000);
          if ( v71 )
          {
            v47 = v71 - 1;
            goto LABEL_60;
          }
        }
        if ( (*(_BYTE *)(a1 + 1072) & 1) != 0 )
        {
          v49 = a1 + 1080;
          v50 = 7;
LABEL_64:
          v51 = v50 & (37 * v18);
          v20 = *(unsigned int *)(v49 + 8LL * v51);
          if ( v18 == (_DWORD)v20 )
          {
LABEL_65:
            v24 |= 0x200u;
          }
          else
          {
            v12 = 1;
            while ( (_DWORD)v20 != -1 )
            {
              a5 = (unsigned int)(v12 + 1);
              v51 = v50 & (v12 + v51);
              v20 = *(unsigned int *)(v49 + 8LL * v51);
              if ( v18 == (_DWORD)v20 )
                goto LABEL_65;
              v12 = (unsigned int)a5;
            }
          }
        }
        else
        {
          v70 = *(_DWORD *)(a1 + 1088);
          v49 = *(_QWORD *)(a1 + 1080);
          if ( v70 )
          {
            v50 = v70 - 1;
            goto LABEL_64;
          }
        }
        v52 = *(_DWORD *)(v15 + 36);
        if ( v52 != -3 )
        {
          if ( v52 == -1 )
          {
            if ( v24 > 1 )
            {
LABEL_69:
              v53 = sub_C5F790(v12, v20);
              v54 = v24;
              v55 = v53;
              v56 = *(__m128i **)(v53 + 32);
              if ( *(_QWORD *)(v55 + 24) - (_QWORD)v56 > 0x1Au )
              {
                si128 = _mm_load_si128((const __m128i *)&xmmword_42FF160);
                qmemcpy(&v56[1], "e in a map!", 11);
                *v56 = si128;
                *(_QWORD *)(v55 + 32) += 27LL;
                goto LABEL_71;
              }
              v20 = (__int64)"Unprocessed value in a map!";
              sub_CB6200(v55, "Unprocessed value in a map!", 0x1Bu);
              v54 = v24;
              if ( (v24 & 1) == 0 )
                goto LABEL_72;
              goto LABEL_204;
            }
          }
          else if ( v24 )
          {
            goto LABEL_69;
          }
          goto LABEL_17;
        }
        v91 = v146;
        v20 = *(_QWORD *)a1;
        v92 = *(_QWORD *)(v147 + 40) + 16 * v9;
        LOWORD(v91) = *(_WORD *)v92;
        sub_2FE6CC0((__int64)v161, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), v91, *(_QWORD *)(v92 + 8));
        if ( LOBYTE(v161[0]) )
        {
          v93 = *(_DWORD *)(v147 + 16);
          if ( v93 != 35 && v93 != 9 )
          {
            if ( v24 )
            {
              if ( (v24 & (v24 - 1)) != 0 )
              {
                v111 = sub_C5F790((__int64)v161, v20);
                v54 = v24;
                v55 = v111;
                v112 = *(__m128i **)(v111 + 32);
                if ( *(_QWORD *)(v55 + 24) - (_QWORD)v112 <= 0x16u )
                {
                  v20 = (__int64)"Value in multiple maps!";
                  sub_CB6200(v55, "Value in multiple maps!", 0x17u);
                  v54 = v24;
                }
                else
                {
                  v113 = _mm_load_si128((const __m128i *)&xmmword_42FF1A0);
                  v112[1].m128i_i32[0] = 1634541669;
                  v112[1].m128i_i16[2] = 29552;
                  v112[1].m128i_i8[6] = 33;
                  *v112 = v113;
                  *(_QWORD *)(v55 + 32) += 23LL;
                }
                goto LABEL_71;
              }
              goto LABEL_17;
            }
            if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
            {
LABEL_126:
              v20 = (__int64)v144;
              v94 = 7;
              goto LABEL_127;
            }
LABEL_142:
            v108 = *(_DWORD *)(a1 + 528);
            v20 = *(_QWORD *)(a1 + 520);
            if ( !v108 )
              goto LABEL_237;
            v94 = v108 - 1;
LABEL_127:
            v95 = v94 & (37 * v18);
            v96 = (const char *)(v20 + 24LL * v95);
            v97 = *(unsigned int *)v96;
            if ( (_DWORD)v97 != v18 )
            {
              v118 = 1;
              while ( (_DWORD)v97 != -1 )
              {
                a5 = (unsigned int)(v118 + 1);
                v95 = v94 & (v118 + v95);
                v96 = (const char *)(v20 + 24LL * v95);
                v97 = *(unsigned int *)v96;
                if ( v18 == (_DWORD)v97 )
                  goto LABEL_128;
                v118 = a5;
              }
LABEL_237:
              BUG();
            }
LABEL_128:
            if ( *(_DWORD *)(*((_QWORD *)v96 + 1) + 36LL) == -3 )
            {
              v98 = sub_C5F790(v97, v20);
              v65 = 0;
              v66 = v98;
              v99 = *(__m128i **)(v98 + 32);
              if ( *(_QWORD *)(v66 + 24) - (_QWORD)v99 <= 0x1Eu )
              {
                v20 = (__int64)"Processed value not in any map!";
                sub_CB6200(v66, "Processed value not in any map!", 0x1Fu);
                v65 = v24;
              }
              else
              {
                v100 = _mm_load_si128((const __m128i *)&xmmword_42FF190);
                qmemcpy(&v99[1], "not in any map!", 15);
                *v99 = v100;
                *(_QWORD *)(v66 + 32) += 31LL;
              }
              goto LABEL_80;
            }
            goto LABEL_17;
          }
        }
        if ( v24 > 1 )
        {
          v101 = sub_C5F790((__int64)v161, v20);
          v54 = v24;
          v55 = v101;
          v102 = *(__m128i **)(v101 + 32);
          if ( *(_QWORD *)(v55 + 24) - (_QWORD)v102 <= 0x25u )
          {
            v20 = (__int64)"Value with legal type was transformed!";
            sub_CB6200(v55, "Value with legal type was transformed!", 0x26u);
            v54 = v24;
          }
          else
          {
            v103 = _mm_load_si128((const __m128i *)&xmmword_42FF170);
            v102[2].m128i_i32[0] = 1701671535;
            v102[2].m128i_i16[2] = 8548;
            *v102 = v103;
            v102[1] = _mm_load_si128((const __m128i *)&xmmword_42FF180);
            *(_QWORD *)(v55 + 32) += 38LL;
          }
LABEL_71:
          if ( (v54 & 1) == 0 )
          {
LABEL_72:
            v157 = v54 & 0x80;
            v58 = v54 & 2;
            v59 = v54 & 4;
            v60 = v54 & 8;
            v61 = v54 & 0x10;
            v62 = v54 & 0x20;
            v63 = v54 & 0x40;
            v64 = v54 & 0x100;
            v65 = v54 & 0x200;
            v155 = v64;
            if ( v58 )
            {
              v150 = v65;
              v133 = sub_C5F790(v64, v20);
              v65 = v150;
              v64 = v133;
              v134 = *(__m128i **)(v133 + 32);
              if ( *(_QWORD *)(v64 + 24) - (_QWORD)v134 <= 0x10u )
              {
                v20 = (__int64)" PromotedIntegers";
                sub_CB6200(v64, " PromotedIntegers", 0x11u);
                v65 = v150;
              }
              else
              {
                v135 = _mm_load_si128((const __m128i *)&xmmword_42FF1B0);
                v134[1].m128i_i8[0] = 115;
                *v134 = v135;
                *(_QWORD *)(v64 + 32) += 17LL;
              }
            }
            if ( v59 )
            {
              v152 = v65;
              v139 = sub_C5F790(v64, v20);
              v65 = v152;
              v64 = v139;
              v140 = *(void **)(v139 + 32);
              if ( *(_QWORD *)(v64 + 24) - (_QWORD)v140 <= 0xEu )
              {
                v20 = (__int64)" SoftenedFloats";
                sub_CB6200(v64, " SoftenedFloats", 0xFu);
                v65 = v152;
              }
              else
              {
                qmemcpy(v140, " SoftenedFloats", 15);
                *(_QWORD *)(v64 + 32) += 15LL;
              }
            }
            if ( v60 )
            {
              v148 = v65;
              v128 = sub_C5F790(v64, v20);
              v65 = v148;
              v64 = v128;
              v129 = *(__m128i **)(v128 + 32);
              if ( *(_QWORD *)(v64 + 24) - (_QWORD)v129 <= 0x11u )
              {
                v20 = (__int64)" ScalarizedVectors";
                sub_CB6200(v64, " ScalarizedVectors", 0x12u);
                v65 = v148;
              }
              else
              {
                v130 = _mm_load_si128((const __m128i *)&xmmword_42FF1C0);
                v129[1].m128i_i16[0] = 29554;
                *v129 = v130;
                *(_QWORD *)(v64 + 32) += 18LL;
              }
            }
            if ( v61 )
            {
              v153 = v65;
              v141 = sub_C5F790(v64, v20);
              v65 = v153;
              v64 = v141;
              v142 = *(__m128i **)(v141 + 32);
              if ( *(_QWORD *)(v64 + 24) - (_QWORD)v142 <= 0x10u )
              {
                v20 = (__int64)" ExpandedIntegers";
                sub_CB6200(v64, " ExpandedIntegers", 0x11u);
                v65 = v153;
              }
              else
              {
                v143 = _mm_load_si128((const __m128i *)&xmmword_42FF1D0);
                v142[1].m128i_i8[0] = 115;
                *v142 = v143;
                *(_QWORD *)(v64 + 32) += 17LL;
              }
            }
            if ( v62 )
            {
              v149 = v65;
              v131 = sub_C5F790(v64, v20);
              v65 = v149;
              v64 = v131;
              v132 = *(void **)(v131 + 32);
              if ( *(_QWORD *)(v64 + 24) - (_QWORD)v132 <= 0xEu )
              {
                v20 = (__int64)" ExpandedFloats";
                sub_CB6200(v64, " ExpandedFloats", 0xFu);
                v65 = v149;
              }
              else
              {
                qmemcpy(v132, " ExpandedFloats", 15);
                *(_QWORD *)(v64 + 32) += 15LL;
              }
            }
            if ( v63 )
            {
              v151 = v65;
              v136 = sub_C5F790(v64, v20);
              v65 = v151;
              v137 = v136;
              v138 = *(void **)(v136 + 32);
              if ( *(_QWORD *)(v137 + 24) - (_QWORD)v138 <= 0xCu )
              {
                v20 = (__int64)" SplitVectors";
                sub_CB6200(v137, " SplitVectors", 0xDu);
                v65 = v151;
              }
              else
              {
                qmemcpy(v138, " SplitVectors", 13);
                *(_QWORD *)(v137 + 32) += 13LL;
              }
            }
            v66 = v157;
            if ( v157 )
            {
              v160 = v65;
              v126 = sub_C5F790(v66, v20);
              v65 = v160;
              v66 = v126;
              v127 = *(void **)(v126 + 32);
              if ( *(_QWORD *)(v66 + 24) - (_QWORD)v127 <= 0xEu )
              {
                v20 = (__int64)" WidenedVectors";
                sub_CB6200(v66, " WidenedVectors", 0xFu);
                v65 = v160;
              }
              else
              {
                v20 = 29295;
                qmemcpy(v127, " WidenedVectors", 15);
                *(_QWORD *)(v66 + 32) += 15LL;
              }
            }
            if ( v155 )
            {
              v159 = v65;
              v124 = sub_C5F790(v66, v20);
              v65 = v159;
              v66 = v124;
              v125 = *(void **)(v124 + 32);
              if ( *(_QWORD *)(v66 + 24) - (_QWORD)v125 <= 0xEu )
              {
                v20 = (__int64)" PromotedFloats";
                sub_CB6200(v66, " PromotedFloats", 0xFu);
                v65 = v159;
              }
              else
              {
                qmemcpy(v125, " PromotedFloats", 15);
                *(_QWORD *)(v66 + 32) += 15LL;
              }
            }
LABEL_80:
            if ( v65 )
            {
              v66 = sub_C5F790(v66, v20);
              v122 = *(__m128i **)(v66 + 32);
              if ( *(_QWORD *)(v66 + 24) - (_QWORD)v122 <= 0x10u )
              {
                v20 = (__int64)" SoftPromoteHalfs";
                sub_CB6200(v66, " SoftPromoteHalfs", 0x11u);
              }
              else
              {
                v123 = _mm_load_si128((const __m128i *)&xmmword_45242E0);
                v122[1].m128i_i8[0] = 115;
                *v122 = v123;
                *(_QWORD *)(v66 + 32) += 17LL;
              }
            }
            v67 = sub_C5F790(v66, v20);
            v68 = *(_BYTE **)(v67 + 32);
            if ( *(_BYTE **)(v67 + 24) == v68 )
            {
              sub_CB6200(v67, (unsigned __int8 *)"\n", 1u);
            }
            else
            {
              *v68 = 10;
              ++*(_QWORD *)(v67 + 32);
            }
            BUG();
          }
LABEL_204:
          v158 = v54;
          v119 = sub_C5F790(v55, v20);
          v54 = v158;
          v120 = v119;
          v121 = *(void **)(v119 + 32);
          if ( *(_QWORD *)(v120 + 24) - (_QWORD)v121 <= 0xEu )
          {
            v20 = (__int64)" ReplacedValues";
            sub_CB6200(v120, " ReplacedValues", 0xFu);
            v54 = v158;
          }
          else
          {
            qmemcpy(v121, " ReplacedValues", 15);
            *(_QWORD *)(v120 + 32) += 15LL;
          }
          goto LABEL_72;
        }
LABEL_17:
        ++v9;
      }
      while ( v9 != v156 );
LABEL_18:
      v147 = *(_QWORD *)(v147 + 8);
      if ( v145 == v147 )
      {
        if ( v162 != v164 )
          _libc_free((unsigned __int64)v162);
        return;
      }
    }
  }
}
