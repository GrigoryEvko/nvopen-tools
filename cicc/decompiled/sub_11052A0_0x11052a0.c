// Function: sub_11052A0
// Address: 0x11052a0
//
unsigned __int8 *__fastcall sub_11052A0(const __m128i *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  __int64 v7; // r15
  __int64 v8; // rdx
  unsigned __int16 v9; // bx
  int v10; // eax
  int v11; // eax
  unsigned __int8 *result; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  unsigned int v15; // ebx
  bool v16; // al
  __int64 v17; // rbx
  __int64 v18; // rax
  char v19; // al
  unsigned __int8 *v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int16 v23; // ax
  __int64 *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rbx
  __int64 v28; // r9
  unsigned int **v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rax
  _BYTE *v32; // rax
  int v33; // eax
  bool v34; // al
  __int64 v35; // rbx
  __int64 v36; // r14
  int v37; // eax
  __int64 v38; // r14
  __m128i v39; // rax
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r14
  __int64 *v43; // r10
  unsigned int v44; // eax
  __int64 v45; // rsi
  __m128i v46; // xmm1
  __int64 v47; // rax
  unsigned __int64 v48; // xmm2_8
  __m128i v49; // xmm3
  __int64 v50; // r8
  __int64 v51; // r8
  __int64 v52; // r8
  __int64 v53; // rsi
  int v54; // eax
  __int64 v55; // rcx
  __int64 v56; // r8
  unsigned int v57; // ebx
  __int64 v58; // r10
  __int16 v59; // ax
  __int64 v60; // rbx
  _BYTE *v61; // rax
  unsigned int v62; // ebx
  int v63; // ebx
  int v64; // eax
  int v65; // eax
  __m128i v66; // rax
  __int64 v67; // rax
  __int64 v68; // r14
  __int64 v69; // rax
  unsigned int v70; // ebx
  unsigned int v71; // ebx
  __int64 v72; // r9
  __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rbx
  __int64 i; // r14
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // r14
  __int64 v81; // rbx
  __int64 v82; // rax
  bool v83; // bl
  unsigned int v84; // edx
  __int64 v85; // rax
  unsigned int v86; // edx
  unsigned int v87; // ebx
  int v88; // eax
  char v89; // al
  __int64 v90; // r9
  __int64 v91; // rdx
  int v92; // r15d
  __int64 *v93; // rbx
  __int64 *v94; // rax
  __int64 v95; // rbx
  __int64 m; // r14
  __int64 v97; // rdx
  unsigned int v98; // esi
  __int64 v99; // rbx
  __int64 k; // r14
  __int64 v101; // rdx
  unsigned int v102; // esi
  __int64 v103; // rbx
  __int64 j; // r14
  __int64 v105; // rdx
  unsigned int v106; // esi
  __int64 v107; // rax
  unsigned int v108; // [rsp+4h] [rbp-FCh]
  __int64 v109; // [rsp+8h] [rbp-F8h]
  __int64 v110; // [rsp+8h] [rbp-F8h]
  unsigned int v111; // [rsp+8h] [rbp-F8h]
  __int64 v112; // [rsp+8h] [rbp-F8h]
  __int64 *v113; // [rsp+10h] [rbp-F0h]
  __int64 v114; // [rsp+10h] [rbp-F0h]
  __int64 *v115; // [rsp+10h] [rbp-F0h]
  __int64 v116; // [rsp+10h] [rbp-F0h]
  int v117; // [rsp+10h] [rbp-F0h]
  __int64 v118; // [rsp+10h] [rbp-F0h]
  __int64 v119; // [rsp+10h] [rbp-F0h]
  unsigned int v120; // [rsp+18h] [rbp-E8h]
  __int64 v121; // [rsp+18h] [rbp-E8h]
  unsigned int v122; // [rsp+18h] [rbp-E8h]
  __int64 *v123; // [rsp+18h] [rbp-E8h]
  unsigned int v124; // [rsp+18h] [rbp-E8h]
  __int64 v125; // [rsp+18h] [rbp-E8h]
  unsigned __int32 v126; // [rsp+18h] [rbp-E8h]
  __int64 v127; // [rsp+18h] [rbp-E8h]
  __int64 v128; // [rsp+18h] [rbp-E8h]
  __int64 v129; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v130; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v131; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v132; // [rsp+18h] [rbp-E8h]
  __int64 v133; // [rsp+18h] [rbp-E8h]
  __int64 v134; // [rsp+18h] [rbp-E8h]
  __int64 v135; // [rsp+18h] [rbp-E8h]
  __int64 v136; // [rsp+18h] [rbp-E8h]
  unsigned int v137; // [rsp+18h] [rbp-E8h]
  __int64 v138; // [rsp+18h] [rbp-E8h]
  __int64 v139; // [rsp+18h] [rbp-E8h]
  __int64 v140; // [rsp+18h] [rbp-E8h]
  __int64 v141; // [rsp+18h] [rbp-E8h]
  __int64 v142; // [rsp+18h] [rbp-E8h]
  __int64 v143; // [rsp+18h] [rbp-E8h]
  __int64 v144; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int32 v145; // [rsp+28h] [rbp-D8h]
  __int64 v146; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int32 v147; // [rsp+38h] [rbp-C8h]
  __int64 v148; // [rsp+40h] [rbp-C0h]
  unsigned int v149; // [rsp+48h] [rbp-B8h]
  int v150[8]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v151; // [rsp+70h] [rbp-90h]
  __m128i v152; // [rsp+80h] [rbp-80h] BYREF
  __m128i v153; // [rsp+90h] [rbp-70h]
  unsigned __int64 v154; // [rsp+A0h] [rbp-60h]
  __int64 v155; // [rsp+A8h] [rbp-58h]
  __m128i v156; // [rsp+B0h] [rbp-50h]
  __int64 v157; // [rsp+C0h] [rbp-40h]

  v6 = *(_QWORD *)(a2 - 32);
  v7 = v6 + 24;
  if ( *(_BYTE *)v6 != 17 )
  {
    v8 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
    if ( (unsigned int)v8 > 1 || *(_BYTE *)v6 > 0x15u || (v32 = sub_AD7630(v6, 0, v8)) == 0 || *v32 != 17 )
    {
LABEL_11:
      v11 = *(_WORD *)(a2 + 2) & 0x3F;
      goto LABEL_12;
    }
    v7 = (__int64)(v32 + 24);
  }
  v8 = *(unsigned int *)(v7 + 8);
  v9 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v9 == 40 )
  {
    if ( (unsigned int)v8 <= 0x40 )
    {
      v34 = *(_QWORD *)v7 == 0;
    }
    else
    {
      v122 = *(_DWORD *)(v7 + 8);
      v33 = sub_C444A0(v7);
      v8 = v122;
      v34 = v122 == v33;
    }
    if ( v34 )
    {
      v35 = *(_QWORD *)(a2 - 64);
      v36 = *(_QWORD *)(v35 + 8);
      v37 = sub_BCB060(v36);
      v38 = sub_AD64C0(v36, (unsigned int)(v37 - 1), 0);
      v123 = (__int64 *)a1[2].m128i_i64[0];
      v39.m128i_i64[0] = (__int64)sub_BD5D20(v35);
      v152 = v39;
      LOWORD(v154) = 773;
      v153.m128i_i64[0] = (__int64)".lobit";
      v40 = sub_F94560(v123, v35, v38, (__int64)&v152, 0);
      v41 = *(_QWORD *)(a3 + 8);
      v42 = v40;
      if ( *(_QWORD *)(v40 + 8) != v41 )
      {
        v43 = (__int64 *)a1[2].m128i_i64[0];
        LOWORD(v154) = 257;
        v113 = v43;
        v124 = sub_BCB060(*(_QWORD *)(v40 + 8));
        v44 = sub_BCB060(v41);
        v42 = sub_10FF770(v113, (unsigned int)(v124 <= v44) + 38, v42, v41, (__int64)&v152, 0, v150[0], 0);
      }
      return sub_F162A0((__int64)a1, a3, v42);
    }
  }
  if ( (unsigned int)v8 <= 0x40 )
  {
    LOBYTE(v8) = *(_QWORD *)v7 == 0;
  }
  else
  {
    v120 = v8;
    v10 = sub_C444A0(v7);
    v8 = v120;
    LOBYTE(v8) = v120 == v10;
  }
  v11 = v9;
  if ( !(_BYTE)v8 )
  {
LABEL_12:
    if ( (unsigned int)(v11 - 32) > 1 )
      return 0;
    v13 = *(_QWORD *)(a2 + 16);
    if ( !v13 || *(_QWORD *)(v13 + 8) )
      return 0;
    v14 = *(_QWORD *)(a2 - 32);
    if ( *(_BYTE *)v14 == 17 )
    {
      v15 = *(_DWORD *)(v14 + 32);
      if ( v15 <= 0x40 )
        v16 = *(_QWORD *)(v14 + 24) == 0;
      else
        v16 = v15 == (unsigned int)sub_C444A0(v14 + 24);
      if ( !v16 )
        return 0;
    }
    else
    {
      v60 = *(_QWORD *)(v14 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v60 + 8) - 17 > 1 || *(_BYTE *)v14 > 0x15u )
        return 0;
      v61 = sub_AD7630(*(_QWORD *)(a2 - 32), 0, v8);
      if ( !v61 || *v61 != 17 )
      {
        if ( *(_BYTE *)(v60 + 8) == 17 )
        {
          v117 = *(_DWORD *)(v60 + 32);
          if ( v117 )
          {
            v83 = 0;
            v84 = 0;
            while ( 1 )
            {
              v137 = v84;
              v85 = sub_AD69F0((unsigned __int8 *)v14, v84);
              if ( !v85 )
                break;
              v86 = v137;
              if ( *(_BYTE *)v85 != 13 )
              {
                if ( *(_BYTE *)v85 != 17 )
                  break;
                v87 = *(_DWORD *)(v85 + 32);
                if ( v87 <= 0x40 )
                {
                  v83 = *(_QWORD *)(v85 + 24) == 0;
                }
                else
                {
                  v88 = sub_C444A0(v85 + 24);
                  v86 = v137;
                  v83 = v87 == v88;
                }
                if ( !v83 )
                  break;
              }
              v84 = v86 + 1;
              if ( v117 == v84 )
              {
                if ( v83 )
                  goto LABEL_19;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v62 = *((_DWORD *)v61 + 8);
      if ( v62 <= 0x40 )
      {
        if ( *((_QWORD *)v61 + 3) )
          return 0;
      }
      else if ( v62 != (unsigned int)sub_C444A0((__int64)(v61 + 24)) )
      {
        return 0;
      }
    }
LABEL_19:
    v17 = *(_QWORD *)(a2 - 64);
    v152.m128i_i64[0] = 0;
    v152.m128i_i64[1] = (__int64)&v146;
    v153.m128i_i64[0] = (__int64)&v144;
    v18 = *(_QWORD *)(v17 + 16);
    if ( !v18 || *(_QWORD *)(v18 + 8) || *(_BYTE *)v17 != 57 )
      return 0;
    v19 = sub_1105100((__int64 **)&v152, 25, *(unsigned __int8 **)(v17 - 64));
    v20 = *(unsigned __int8 **)(v17 - 32);
    if ( v19 && v20 )
    {
      *(_QWORD *)v153.m128i_i64[0] = v20;
    }
    else
    {
      if ( !(unsigned __int8)sub_1105100((__int64 **)&v152, 25, v20) )
        return 0;
      v107 = *(_QWORD *)(v17 - 64);
      if ( !v107 )
        return 0;
      *(_QWORD *)v153.m128i_i64[0] = v107;
    }
    v21 = *(_QWORD *)(a2 - 64);
    v22 = v144;
    v23 = *(_WORD *)(a2 + 2) & 0x3F;
    if ( *(_QWORD *)(v21 + 8) == *(_QWORD *)(a3 + 8) )
    {
      if ( v23 != 32 )
      {
LABEL_27:
        v24 = (__int64 *)a1[2].m128i_i64[0];
        LOWORD(v154) = 257;
        v25 = sub_F94560(v24, v22, v146, (__int64)&v152, 0);
        v26 = a1[2].m128i_i64[0];
        v27 = v25;
        v151 = 257;
        v121 = sub_AD64C0(*(_QWORD *)(v144 + 8), 1, 0);
        v28 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v26 + 80) + 16LL))(
                *(_QWORD *)(v26 + 80),
                28,
                v27,
                v121);
        if ( !v28 )
        {
          LOWORD(v154) = 257;
          v134 = sub_B504D0(28, v27, v121, (__int64)&v152, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(v26 + 88) + 16LL))(
            *(_QWORD *)(v26 + 88),
            v134,
            v150,
            *(_QWORD *)(v26 + 56),
            *(_QWORD *)(v26 + 64));
          v76 = *(_QWORD *)v26;
          v28 = v134;
          for ( i = *(_QWORD *)v26 + 16LL * *(unsigned int *)(v26 + 8); i != v76; v28 = v135 )
          {
            v78 = *(_QWORD *)(v76 + 8);
            v79 = *(_DWORD *)v76;
            v76 += 16;
            v135 = v28;
            sub_B99FD0(v28, v79, v78);
          }
        }
        v29 = (unsigned int **)a1[2].m128i_i64[0];
        v30 = *(_QWORD *)(a3 + 8);
        LOWORD(v154) = 257;
        v31 = sub_A830B0(v29, v28, v30, (__int64)&v152);
        return sub_F162A0((__int64)a1, a3, v31);
      }
    }
    else
    {
      if ( v23 != 32 )
        goto LABEL_27;
      v75 = *(_QWORD *)(*(_QWORD *)(v21 + 32LL * (*(_QWORD *)(v21 - 64) == v144) - 64) + 16LL);
      if ( !v75 || *(_QWORD *)(v75 + 8) )
        return 0;
    }
    v73 = a1[2].m128i_i64[0];
    v151 = 257;
    v133 = sub_AD62B0(*(_QWORD *)(v144 + 8));
    v74 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v73 + 80) + 16LL))(
            *(_QWORD *)(v73 + 80),
            30,
            v22,
            v133);
    if ( !v74 )
    {
      LOWORD(v154) = 257;
      v142 = sub_B504D0(30, v22, v133, (__int64)&v152, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(v73 + 88) + 16LL))(
        *(_QWORD *)(v73 + 88),
        v142,
        v150,
        *(_QWORD *)(v73 + 56),
        *(_QWORD *)(v73 + 64));
      v103 = *(_QWORD *)v73;
      v74 = v142;
      for ( j = *(_QWORD *)v73 + 16LL * *(unsigned int *)(v73 + 8); j != v103; v74 = v143 )
      {
        v105 = *(_QWORD *)(v103 + 8);
        v106 = *(_DWORD *)v103;
        v103 += 16;
        v143 = v74;
        sub_B99FD0(v74, v106, v105);
      }
    }
    v144 = v74;
    v22 = v74;
    goto LABEL_27;
  }
  if ( (unsigned int)v9 - 32 > 1 )
    return 0;
  v45 = *(_QWORD *)(a2 - 64);
  v46 = _mm_loadu_si128(a1 + 7);
  v47 = a1[10].m128i_i64[0];
  v48 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v152 = _mm_loadu_si128(a1 + 6);
  v49 = _mm_loadu_si128(a1 + 9);
  v157 = v47;
  v154 = v48;
  v153 = v46;
  v155 = a3;
  v156 = v49;
  sub_9AC330((__int64)&v146, v45, 0, &v152);
  v8 = v147;
  v152.m128i_i32[2] = v147;
  if ( v147 <= 0x40 )
  {
    v50 = v146;
LABEL_40:
    v51 = ~v50;
    if ( !(_DWORD)v8 )
      goto LABEL_52;
    v52 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v8) & v51;
    v145 = v8;
    v144 = v52;
    goto LABEL_42;
  }
  sub_C43780((__int64)&v152, (const void **)&v146);
  v8 = v152.m128i_u32[2];
  if ( v152.m128i_i32[2] <= 0x40u )
  {
    v50 = v152.m128i_i64[0];
    goto LABEL_40;
  }
  sub_C43D10((__int64)&v152);
  v8 = v152.m128i_u32[2];
  v52 = v152.m128i_i64[0];
  v145 = v152.m128i_u32[2];
  v144 = v152.m128i_i64[0];
  if ( v152.m128i_i32[2] > 0x40u )
  {
    v110 = v152.m128i_i64[0];
    v126 = v152.m128i_u32[2];
    v63 = v126 - sub_C444A0((__int64)&v144);
    v64 = sub_C44630((__int64)&v144);
    v8 = v126;
    v56 = v110;
    if ( v64 != 1
      || (v127 = v110, v111 = v8, v114 = *(_QWORD *)(a3 + 8), v65 = sub_BCB060(v114), v56 = v127, v65 == v63) )
    {
LABEL_50:
      if ( v56 )
        j_j___libc_free_0_0(v56);
      goto LABEL_52;
    }
    v55 = v114;
    v8 = v111;
    v57 = v63 - 1;
    goto LABEL_46;
  }
LABEL_42:
  if ( !v52 )
    goto LABEL_52;
  if ( (v52 & (v52 - 1)) != 0 )
    goto LABEL_52;
  _BitScanReverse64((unsigned __int64 *)&v53, v52);
  v108 = v8;
  v109 = v52;
  v125 = *(_QWORD *)(a3 + 8);
  v54 = sub_BCB060(v125);
  v55 = v125;
  v56 = v109;
  v8 = v108;
  if ( 64 - ((unsigned int)v53 ^ 0x3F) == v54 )
    goto LABEL_52;
  v57 = 63 - (v53 ^ 0x3F);
LABEL_46:
  v58 = *(_QWORD *)(a2 - 64);
  if ( *(_QWORD *)(v58 + 8) == v55 || (v59 = *(_WORD *)(a2 + 2) & 0x3F, v59 == 33) )
  {
    if ( v57 )
    {
      v128 = *(_QWORD *)(a2 - 64);
      v115 = (__int64 *)a1[2].m128i_i64[0];
      v66.m128i_i64[0] = (__int64)sub_BD5D20(v128);
      v152 = v66;
      LOWORD(v154) = 773;
      v153.m128i_i64[0] = (__int64)".lobit";
      v67 = sub_AD64C0(*(_QWORD *)(v128 + 8), v57, 0);
      v58 = sub_F94560(v115, v128, v67, (__int64)&v152, 0);
    }
    v59 = *(_WORD *)(a2 + 2) & 0x3F;
  }
  else if ( v57 )
  {
    if ( (unsigned int)v8 > 0x40 )
      goto LABEL_50;
LABEL_52:
    if ( v149 > 0x40 && v148 )
      j_j___libc_free_0_0(v148);
    if ( v147 > 0x40 && v146 )
      j_j___libc_free_0_0(v146);
    goto LABEL_11;
  }
  if ( v59 == 32 )
  {
    v80 = a1[2].m128i_i64[0];
    v151 = 257;
    v136 = v58;
    v81 = sub_AD64C0(*(_QWORD *)(v58 + 8), 1, 0);
    v82 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v80 + 80) + 16LL))(
            *(_QWORD *)(v80 + 80),
            30,
            v136,
            v81);
    if ( !v82 )
    {
      LOWORD(v154) = 257;
      v140 = sub_B504D0(30, v136, v81, (__int64)&v152, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(v80 + 88) + 16LL))(
        *(_QWORD *)(v80 + 88),
        v140,
        v150,
        *(_QWORD *)(v80 + 56),
        *(_QWORD *)(v80 + 64));
      v99 = *(_QWORD *)v80;
      v82 = v140;
      for ( k = *(_QWORD *)v80 + 16LL * *(unsigned int *)(v80 + 8); k != v99; v82 = v141 )
      {
        v101 = *(_QWORD *)(v99 + 8);
        v102 = *(_DWORD *)v99;
        v99 += 16;
        v141 = v82;
        sub_B99FD0(v82, v102, v101);
      }
    }
    v58 = v82;
  }
  v68 = *(_QWORD *)(a3 + 8);
  if ( *(_QWORD *)(v58 + 8) == v68 )
  {
    result = sub_F162A0((__int64)a1, a3, v58);
  }
  else
  {
    v69 = a1[2].m128i_i64[0];
    v112 = v58;
    v151 = 257;
    v129 = v69;
    v116 = *(_QWORD *)(v58 + 8);
    v70 = sub_BCB060(v116);
    v71 = (v70 <= (unsigned int)sub_BCB060(v68)) + 38;
    if ( v68 == v116 )
    {
      v72 = v112;
    }
    else
    {
      v72 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(v129 + 80) + 120LL))(
              *(_QWORD *)(v129 + 80),
              v71,
              v112,
              v68);
      if ( !v72 )
      {
        LOWORD(v154) = 257;
        v118 = sub_B51D30(v71, v112, v68, (__int64)&v152, 0, 0);
        v89 = sub_920620(v118);
        v90 = v118;
        if ( v89 )
        {
          v91 = *(_QWORD *)(v129 + 96);
          v92 = *(_DWORD *)(v129 + 104);
          if ( v91 )
          {
            sub_B99FD0(v118, 3u, v91);
            v90 = v118;
          }
          v119 = v90;
          sub_B45150(v90, v92);
          v90 = v119;
        }
        v93 = (__int64 *)v129;
        v138 = v90;
        (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v93[11] + 16LL))(
          v93[11],
          v90,
          v150,
          v93[7],
          v93[8]);
        v94 = v93;
        v95 = *v93;
        v72 = v138;
        for ( m = v95 + 16LL * *((unsigned int *)v94 + 2); m != v95; v72 = v139 )
        {
          v97 = *(_QWORD *)(v95 + 8);
          v98 = *(_DWORD *)v95;
          v95 += 16;
          v139 = v72;
          sub_B99FD0(v72, v98, v97);
        }
      }
    }
    result = sub_F162A0((__int64)a1, a3, v72);
  }
  if ( v145 > 0x40 && v144 )
  {
    v130 = result;
    j_j___libc_free_0_0(v144);
    result = v130;
  }
  if ( v149 > 0x40 && v148 )
  {
    v131 = result;
    j_j___libc_free_0_0(v148);
    result = v131;
  }
  if ( v147 > 0x40 && v146 )
  {
    v132 = result;
    j_j___libc_free_0_0(v146);
    return v132;
  }
  return result;
}
