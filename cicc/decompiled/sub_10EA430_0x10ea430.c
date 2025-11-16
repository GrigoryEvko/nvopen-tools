// Function: sub_10EA430
// Address: 0x10ea430
//
unsigned __int8 *__fastcall sub_10EA430(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // rdx
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  _QWORD *v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  char v11; // al
  int v12; // r9d
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v26; // rax
  int v27; // edx
  int v28; // esi
  int v29; // ecx
  char v30; // cl
  __int64 v31; // rbx
  __int64 v32; // r15
  unsigned __int8 *v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdi
  unsigned __int8 *v38; // r13
  __int64 v39; // r15
  unsigned __int8 *v40; // rax
  __m128i v41; // xmm6
  __m128i v42; // xmm0
  __m128i v43; // xmm2
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  unsigned int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rsi
  __int64 v56; // r10
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r10
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rbx
  unsigned int v64; // r15d
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rdi
  unsigned __int8 *v68; // r13
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rsi
  __int64 v72; // r8
  __int64 v73; // rax
  __int16 v74; // ax
  unsigned int v75; // edx
  unsigned int v76; // r15d
  unsigned __int8 *v77; // rcx
  __int64 v78; // rax
  int v79; // eax
  __int64 v80; // rax
  int v81; // eax
  __int64 v82; // rax
  int v83; // eax
  __int64 v84; // rax
  __int16 v85; // ax
  __int64 v86; // rax
  int v87; // eax
  __int64 v88; // rax
  int v89; // eax
  __int64 v90; // rax
  int v91; // eax
  __int64 v92; // rax
  __int16 v93; // ax
  __int64 v94; // rax
  __int16 v95; // ax
  __int64 v96; // rax
  __int16 v97; // ax
  __int64 v98; // rax
  __int16 v99; // ax
  unsigned int v100; // [rsp+4h] [rbp-BCh]
  char v101; // [rsp+8h] [rbp-B8h]
  unsigned int v102; // [rsp+8h] [rbp-B8h]
  __int64 v103; // [rsp+8h] [rbp-B8h]
  __int64 v104; // [rsp+8h] [rbp-B8h]
  __int64 v105; // [rsp+8h] [rbp-B8h]
  __int64 v106; // [rsp+8h] [rbp-B8h]
  __int64 v107; // [rsp+8h] [rbp-B8h]
  __int64 v108; // [rsp+8h] [rbp-B8h]
  __int64 v109; // [rsp+8h] [rbp-B8h]
  __int64 v110; // [rsp+8h] [rbp-B8h]
  __int64 v111; // [rsp+8h] [rbp-B8h]
  __int64 v112; // [rsp+8h] [rbp-B8h]
  __int64 v113; // [rsp+8h] [rbp-B8h]
  __int16 v115; // [rsp+18h] [rbp-A8h]
  int v116; // [rsp+18h] [rbp-A8h]
  int v117; // [rsp+18h] [rbp-A8h]
  __int64 v118; // [rsp+18h] [rbp-A8h]
  int v119; // [rsp+20h] [rbp-A0h]
  int v120; // [rsp+20h] [rbp-A0h]
  int v121; // [rsp+20h] [rbp-A0h]
  int v122; // [rsp+20h] [rbp-A0h]
  __int64 v123; // [rsp+20h] [rbp-A0h]
  int v124; // [rsp+20h] [rbp-A0h]
  __int64 v125; // [rsp+20h] [rbp-A0h]
  unsigned int v126; // [rsp+20h] [rbp-A0h]
  int v127; // [rsp+20h] [rbp-A0h]
  __int64 v128; // [rsp+30h] [rbp-90h] BYREF
  __int64 v129; // [rsp+38h] [rbp-88h]
  __m128i v130[2]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v131; // [rsp+60h] [rbp-60h]
  __int64 v132; // [rsp+68h] [rbp-58h]
  __m128i v133; // [rsp+70h] [rbp-50h]
  __int64 v134; // [rsp+80h] [rbp-40h]

  v2 = a2;
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v6 = *(_QWORD *)(a2 + 32 * (1 - v4));
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v115 = (__int16)v7;
  v119 = (int)v7;
  v130[0].m128i_i64[0] = *(_QWORD *)(sub_B43CB0(a2) + 120);
  v101 = sub_A73ED0(v130, 72);
  v130[0].m128i_i64[0] = (__int64)&v128;
  v11 = sub_995E90(v130, v5, v8, v9, v10);
  v12 = v119;
  if ( v11 )
  {
    v13 = sub_C65050(v115);
    v14 = sub_AD64C0(*(_QWORD *)(v6 + 8), v13, 0);
    v15 = a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_QWORD *)v15 )
    {
      v16 = *(_QWORD *)(v15 + 8);
      **(_QWORD **)(v15 + 16) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(v15 + 16);
    }
    *(_QWORD *)v15 = v14;
    if ( v14 )
    {
      v17 = *(_QWORD *)(v14 + 16);
      *(_QWORD *)(v15 + 8) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = v15 + 8;
      *(_QWORD *)(v15 + 16) = v14 + 16;
      *(_QWORD *)(v14 + 16) = v15;
    }
    v18 = v128;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v19 = *(_QWORD *)(a2 - 8);
    else
      v19 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v20 = *(_QWORD *)v19;
    if ( *(_QWORD *)v19 )
    {
      v21 = *(_QWORD *)(v19 + 8);
      **(_QWORD **)(v19 + 16) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = *(_QWORD *)(v19 + 16);
    }
    *(_QWORD *)v19 = v18;
    if ( v18 )
    {
      v22 = *(_QWORD *)(v18 + 16);
      *(_QWORD *)(v19 + 8) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = v19 + 8;
      *(_QWORD *)(v19 + 16) = v18 + 16;
      *(_QWORD *)(v18 + 16) = v19;
    }
LABEL_20:
    if ( *(_BYTE *)v20 > 0x1Cu )
    {
      v130[0].m128i_i64[0] = v20;
      v23 = a1[2].m128i_i64[1] + 2096;
      sub_10E8740(v23, v130[0].m128i_i64);
      v24 = *(_QWORD *)(v20 + 16);
      if ( v24 )
      {
        if ( !*(_QWORD *)(v24 + 8) )
        {
          v130[0].m128i_i64[0] = *(_QWORD *)(v24 + 24);
          sub_10E8740(v23, v130[0].m128i_i64);
        }
      }
    }
    return (unsigned __int8 *)v2;
  }
  if ( *(_BYTE *)v5 == 85 )
  {
    v26 = *(_QWORD *)(v5 - 32);
    if ( v26 )
    {
      if ( !*(_BYTE *)v26 && *(_QWORD *)(v26 + 24) == *(_QWORD *)(v5 + 80) && *(_DWORD *)(v26 + 36) == 170 )
      {
        v123 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
        if ( v123 )
        {
          v51 = sub_C650C0(v115);
          v52 = sub_AD64C0(*(_QWORD *)(v6 + 8), v51, 0);
          v53 = a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          if ( *(_QWORD *)v53 )
          {
            v54 = *(_QWORD *)(v53 + 8);
            **(_QWORD **)(v53 + 16) = v54;
            if ( v54 )
              *(_QWORD *)(v54 + 16) = *(_QWORD *)(v53 + 16);
          }
          *(_QWORD *)v53 = v52;
          if ( v52 )
          {
            v55 = *(_QWORD *)(v52 + 16);
            *(_QWORD *)(v53 + 8) = v55;
            if ( v55 )
              *(_QWORD *)(v55 + 16) = v53 + 8;
            *(_QWORD *)(v53 + 16) = v52 + 16;
            *(_QWORD *)(v52 + 16) = v53;
          }
          if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
            v56 = *(_QWORD *)(a2 - 8);
          else
            v56 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
          v20 = *(_QWORD *)v56;
          if ( *(_QWORD *)v56 )
          {
            v57 = *(_QWORD *)(v56 + 8);
            **(_QWORD **)(v56 + 16) = v57;
            if ( v57 )
              *(_QWORD *)(v57 + 16) = *(_QWORD *)(v56 + 16);
          }
          *(_QWORD *)v56 = v123;
          v58 = *(_QWORD *)(v123 + 16);
          *(_QWORD *)(v56 + 8) = v58;
          if ( v58 )
            *(_QWORD *)(v58 + 16) = v56 + 8;
          *(_QWORD *)(v56 + 16) = v123 + 16;
          *(_QWORD *)(v123 + 16) = v56;
          goto LABEL_20;
        }
      }
    }
  }
  v27 = v12 & 3;
  v28 = v12 & 0x3FC;
  v29 = v28 ^ 0x3FC;
  if ( v28 != 516 && v29 != 516 )
  {
    if ( v28 != 512 && v28 != 4 )
    {
      if ( (v29 == 512 || v29 == 4) && (v27 == 0 || v27 == 3) && !v101 )
      {
        v122 = v12 & 3;
        v48 = sub_AD9500(*(_QWORD *)(v5 + 8), v29 == 4);
        LOWORD(v131) = 257;
        HIDWORD(v129) = 0;
        v49 = a1[2].m128i_i64[0];
        if ( v122 == 3 )
          v50 = sub_B35C90(v49, 0xEu, v5, v48, (__int64)v130, 0, v129, 0);
        else
          v50 = sub_B35C90(v49, 6u, v5, v48, (__int64)v130, 0, v129, 0);
        v38 = (unsigned __int8 *)v50;
        goto LABEL_43;
      }
      goto LABEL_33;
    }
    if ( v27 != 3 && (v12 & 3) != 0 )
      goto LABEL_33;
    if ( !v101 )
    {
      v120 = v12 & 3;
      v36 = sub_AD9500(*(_QWORD *)(v5 + 8), v28 == 4);
      HIDWORD(v129) = 0;
      LOWORD(v131) = 257;
      v37 = a1[2].m128i_i64[0];
      if ( v120 == 3 )
        v38 = (unsigned __int8 *)sub_B35C90(v37, 9u, v5, v36, (__int64)v130, 0, v129, 0);
      else
        v38 = (unsigned __int8 *)sub_B35C90(v37, 1u, v5, v36, (__int64)v130, 0, v129, 0);
LABEL_43:
      sub_BD6B90(v38, (unsigned __int8 *)a2);
      return sub_F162A0((__int64)a1, a2, (__int64)v38);
    }
LABEL_49:
    if ( v29 != 4 && v29 != 512 )
      goto LABEL_60;
LABEL_33:
    v30 = v101 ^ 1;
    if ( v12 == 3 && v30 )
    {
      v31 = (__int64)a1;
      LOWORD(v131) = 257;
      v32 = a1[2].m128i_i64[0];
      v33 = sub_AD9290(*(_QWORD *)(v5 + 8), 0);
      HIDWORD(v129) = 0;
      v34 = (unsigned __int8 *)sub_B35C90(v32, 8u, v5, (__int64)v33, (__int64)v130, 0, (unsigned int)v129, 0);
LABEL_36:
      v35 = (__int64)v34;
      sub_BD6B90(v34, (unsigned __int8 *)a2);
      return sub_F162A0(v31, a2, v35);
    }
    if ( v12 == 1020 && v30 )
    {
      v31 = (__int64)a1;
      LOWORD(v131) = 257;
      v39 = a1[2].m128i_i64[0];
      v40 = sub_AD9290(*(_QWORD *)(v5 + 8), 0);
      HIDWORD(v129) = 0;
      v34 = (unsigned __int8 *)sub_B35C90(v39, 7u, v5, (__int64)v40, (__int64)v130, 0, (unsigned int)v129, 0);
      goto LABEL_36;
    }
    if ( v101 || v27 != 3 && (v12 & 3) != 0 )
      goto LABEL_60;
    v100 = v12 & 3;
    v102 = v12 & 0x3FC;
    v117 = v12;
    v125 = *(_QWORD *)(v5 + 8);
    v69 = sub_B43CB0(a2);
    v71 = v102;
    v72 = v125;
    v12 = v117;
    if ( v102 != 768 )
    {
      if ( v102 <= 0x300 )
      {
        if ( v102 == 124 )
        {
          if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
            v72 = **(_QWORD **)(v125 + 16);
          v113 = v69;
          v98 = sub_BCAC60(v72, v71, v100, v70, v72);
          v99 = sub_B2DB90(v113, v98);
          v12 = v117;
          v75 = v100;
          if ( HIBYTE(v99) )
            goto LABEL_60;
        }
        else
        {
          if ( v102 <= 0x7C )
          {
            switch ( v102 )
            {
              case 0x1Cu:
                if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
                  v72 = **(_QWORD **)(v125 + 16);
                v112 = v69;
                v96 = sub_BCAC60(v72, v71, v100, v70, v72);
                v97 = sub_B2DB90(v112, v96);
                v12 = v117;
                v75 = v100;
                if ( HIBYTE(v97) )
                  goto LABEL_60;
                break;
              case 0x60u:
                if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
                  v72 = **(_QWORD **)(v125 + 16);
                v103 = v69;
                v73 = sub_BCAC60(v72, v71, v100, v70, v72);
                v74 = sub_B2DB90(v103, v73);
                v12 = v117;
                v75 = v100;
                if ( !HIBYTE(v74) )
                  goto LABEL_105;
                goto LABEL_60;
              case 0xCu:
                if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
                  v72 = **(_QWORD **)(v125 + 16);
                v104 = v69;
                v78 = sub_BCAC60(v72, v71, v100, v70, v72);
                v79 = sub_B2DB90(v104, v78);
                v12 = v117;
                v75 = v100;
                if ( (unsigned __int8)(BYTE1(v79) - 1) > 1u )
                  goto LABEL_60;
                break;
              default:
                goto LABEL_60;
            }
            v76 = 4;
            goto LABEL_106;
          }
          if ( v102 == 240 )
          {
            if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
              v72 = **(_QWORD **)(v125 + 16);
            v127 = v117;
            v118 = v69;
            v88 = sub_BCAC60(v72, v102, v100, v70, v72);
            v89 = sub_B2DB90(v118, v88);
            v12 = v127;
            if ( (unsigned __int8)(BYTE1(v89) - 1) <= 1u )
            {
              v75 = v100;
LABEL_105:
              v76 = 1;
LABEL_106:
              v126 = v75;
              v77 = sub_AD9290(*(_QWORD *)(v5 + 8), 0);
              v31 = (__int64)a1;
              LOWORD(v131) = 257;
              HIDWORD(v129) = 0;
              if ( v126 == 3 )
                v76 |= 8u;
              v34 = (unsigned __int8 *)sub_B35C90(a1[2].m128i_i64[0], v76, v5, (__int64)v77, (__int64)v130, 0, v129, 0);
              goto LABEL_36;
            }
LABEL_60:
            v41 = _mm_loadu_si128(a1 + 9);
            v42 = _mm_loadu_si128(a1 + 6);
            v43 = _mm_loadu_si128(a1 + 7);
            v44 = a1[10].m128i_i64[0];
            v131 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
            v121 = v12;
            v134 = v44;
            v132 = a2;
            v130[0] = v42;
            v130[1] = v43;
            v133 = v41;
            v45 = sub_9B4030((__int64 *)v5, v12, 0, v130);
            v46 = v45 & (unsigned int)v121;
            if ( v121 == (_DWORD)v46 )
            {
              v2 = 0;
              if ( v45 == v121 )
              {
                v47 = sub_AD64C0(*(_QWORD *)(a2 + 8), 1, 0);
                return sub_F162A0((__int64)a1, a2, v47);
              }
            }
            else
            {
              v59 = sub_AD64C0(*(_QWORD *)(v6 + 8), v46, 0);
              v60 = a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
              if ( *(_QWORD *)v60 )
              {
                v61 = *(_QWORD *)(v60 + 8);
                **(_QWORD **)(v60 + 16) = v61;
                if ( v61 )
                  *(_QWORD *)(v61 + 16) = *(_QWORD *)(v60 + 16);
              }
              *(_QWORD *)v60 = v59;
              if ( v59 )
              {
                v62 = *(_QWORD *)(v59 + 16);
                *(_QWORD *)(v60 + 8) = v62;
                if ( v62 )
                  *(_QWORD *)(v62 + 16) = v60 + 8;
                *(_QWORD *)(v60 + 16) = v59 + 16;
                *(_QWORD *)(v59 + 16) = v60;
              }
            }
            return (unsigned __int8 *)v2;
          }
          if ( v102 != 252 )
            goto LABEL_60;
          if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
            v72 = **(_QWORD **)(v125 + 16);
          v105 = v69;
          v80 = sub_BCAC60(v72, v71, v100, v70, v72);
          v81 = sub_B2DB90(v105, v80);
          v12 = v117;
          v75 = v100;
          if ( (unsigned __int8)(BYTE1(v81) - 1) > 1u )
            goto LABEL_60;
        }
        v76 = 5;
        goto LABEL_106;
      }
      if ( v102 == 924 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
          v72 = **(_QWORD **)(v125 + 16);
        v111 = v69;
        v94 = sub_BCAC60(v72, v71, v100, v70, v72);
        v95 = sub_B2DB90(v111, v94);
        v12 = v117;
        v75 = v100;
        if ( HIBYTE(v95) )
          goto LABEL_60;
      }
      else
      {
        if ( v102 > 0x39C )
        {
          if ( v102 == 992 )
          {
            if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
              v72 = **(_QWORD **)(v125 + 16);
            v110 = v69;
            v92 = sub_BCAC60(v72, v71, v100, v70, v72);
            v93 = sub_B2DB90(v110, v92);
            v12 = v117;
            v75 = v100;
            if ( HIBYTE(v93) )
              goto LABEL_60;
          }
          else
          {
            if ( v102 != 1008 )
              goto LABEL_60;
            if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
              v72 = **(_QWORD **)(v125 + 16);
            v106 = v69;
            v82 = sub_BCAC60(v72, v71, v100, v70, v72);
            v83 = sub_B2DB90(v106, v82);
            v12 = v117;
            v75 = v100;
            if ( (unsigned __int8)(BYTE1(v83) - 1) > 1u )
              goto LABEL_60;
          }
          v76 = 3;
          goto LABEL_106;
        }
        if ( v102 != 780 )
        {
          if ( v102 != 896 )
            goto LABEL_60;
          if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
            v72 = **(_QWORD **)(v125 + 16);
          v107 = v69;
          v84 = sub_BCAC60(v72, v71, v100, v70, v72);
          v85 = sub_B2DB90(v107, v84);
          v12 = v117;
          v75 = v100;
          if ( HIBYTE(v85) )
            goto LABEL_60;
LABEL_133:
          v76 = 2;
          goto LABEL_106;
        }
        if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
          v72 = **(_QWORD **)(v125 + 16);
        v108 = v69;
        v86 = sub_BCAC60(v72, v71, v100, v70, v72);
        v87 = sub_B2DB90(v108, v86);
        v12 = v117;
        v75 = v100;
        if ( (unsigned __int8)(BYTE1(v87) - 1) > 1u )
          goto LABEL_60;
      }
      v76 = 6;
      goto LABEL_106;
    }
    if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
      v72 = **(_QWORD **)(v125 + 16);
    v109 = v69;
    v90 = sub_BCAC60(v72, v71, v100, v70, v72);
    v91 = sub_B2DB90(v109, v90);
    v12 = v117;
    v75 = v100;
    if ( (unsigned __int8)(BYTE1(v91) - 1) > 1u )
      goto LABEL_60;
    goto LABEL_133;
  }
  if ( v27 != 3 && (v12 & 3) != 0 )
    goto LABEL_33;
  if ( v101 )
  {
    if ( v28 != 4 && v28 != 512 )
      goto LABEL_33;
    goto LABEL_49;
  }
  v116 = v12 & 3;
  v124 = v12 & 0x3FC ^ 0x3FC;
  v63 = sub_AD9500(*(_QWORD *)(v5 + 8), 0);
  v64 = 8 * (v116 == 3) + 1;
  if ( v124 == 516 )
    v64 = 8 * (v116 == 3) + 6;
  HIDWORD(v129) = 0;
  v65 = a1[2].m128i_i64[0];
  LOWORD(v131) = 257;
  v66 = sub_B33BC0(v65, 0xAAu, v5, (unsigned int)v129, (__int64)v130);
  HIDWORD(v129) = 0;
  v67 = a1[2].m128i_i64[0];
  LOWORD(v131) = 257;
  v68 = (unsigned __int8 *)sub_B35C90(v67, v64, v66, v63, (__int64)v130, 0, (unsigned int)v129, 0);
  sub_BD6B90(v68, (unsigned __int8 *)a2);
  return sub_F162A0((__int64)a1, a2, (__int64)v68);
}
