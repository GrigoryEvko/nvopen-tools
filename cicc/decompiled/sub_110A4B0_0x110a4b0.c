// Function: sub_110A4B0
// Address: 0x110a4b0
//
unsigned __int8 *__fastcall sub_110A4B0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v8; // rax
  unsigned __int8 *result; // rax
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rax
  __m128i v14; // xmm1
  unsigned __int64 v15; // xmm2_8
  __m128i v16; // xmm3
  unsigned __int8 v17; // al
  _BYTE *v18; // rax
  unsigned int v19; // r15d
  __int64 v20; // rax
  _BYTE *v21; // rax
  _BYTE *v22; // rax
  int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rax
  _BYTE *v26; // r11
  __int64 v27; // rax
  unsigned int **v28; // r12
  _BYTE *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // rdx
  _BYTE *v39; // r13
  __int64 *v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 *v43; // rdi
  __int64 v44; // r14
  __int64 v45; // rcx
  __int64 v46; // rax
  _BYTE *v47; // rdx
  unsigned __int8 v48; // al
  unsigned __int8 *v49; // r13
  unsigned int v50; // eax
  __int64 v51; // rax
  unsigned __int8 *v52; // r13
  unsigned int v53; // eax
  __int64 v54; // rax
  _BYTE *v55; // rax
  _BYTE *v56; // rax
  __int64 v57; // rax
  __int64 v58; // r14
  __int64 v59; // r13
  const char *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r15
  __int64 v63; // r11
  __int64 v64; // rdi
  unsigned int v65; // eax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r15
  __int64 v70; // r12
  __int64 v71; // r14
  __int64 v72; // r8
  __int64 v73; // rsi
  unsigned int **v74; // r12
  _BYTE *v75; // rax
  __int64 v76; // rax
  __int64 v77; // r13
  _BYTE *v78; // r14
  __int64 v79; // rax
  __int64 v80; // r11
  __int64 v81; // r12
  __int64 v82; // rbx
  __int64 v83; // r13
  __int64 v84; // rdx
  unsigned int v85; // esi
  __int64 v86; // rbx
  __int64 v87; // r13
  __int64 v88; // rdx
  unsigned int v89; // esi
  __int64 v90; // r12
  __int64 v91; // r14
  __int64 v92; // rdx
  unsigned int v93; // esi
  _BYTE *v94; // rax
  unsigned int **v95; // rdi
  __int64 v96; // r12
  __int64 v97; // rax
  _BYTE *v98; // rax
  int v99; // eax
  int v100; // [rsp+4h] [rbp-11Ch]
  __int64 v101; // [rsp+8h] [rbp-118h]
  __int64 v102; // [rsp+8h] [rbp-118h]
  unsigned int v103; // [rsp+8h] [rbp-118h]
  __int64 v104; // [rsp+10h] [rbp-110h]
  _BYTE *v105; // [rsp+10h] [rbp-110h]
  unsigned __int8 *v106; // [rsp+10h] [rbp-110h]
  int v107; // [rsp+10h] [rbp-110h]
  unsigned int v108; // [rsp+10h] [rbp-110h]
  __int64 v109; // [rsp+10h] [rbp-110h]
  int v110; // [rsp+18h] [rbp-108h]
  int v111; // [rsp+18h] [rbp-108h]
  unsigned __int8 *v112; // [rsp+18h] [rbp-108h]
  __int64 v113; // [rsp+18h] [rbp-108h]
  int v114; // [rsp+28h] [rbp-F8h]
  __int64 v115; // [rsp+28h] [rbp-F8h]
  _BYTE *v116; // [rsp+28h] [rbp-F8h]
  __int64 v117; // [rsp+28h] [rbp-F8h]
  __int64 v118; // [rsp+28h] [rbp-F8h]
  _BYTE *v119; // [rsp+38h] [rbp-E8h] BYREF
  __int64 v120[4]; // [rsp+40h] [rbp-E0h] BYREF
  __int16 v121; // [rsp+60h] [rbp-C0h]
  _QWORD v122[4]; // [rsp+70h] [rbp-B0h] BYREF
  __int16 v123; // [rsp+90h] [rbp-90h]
  __m128i v124[2]; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v125; // [rsp+C0h] [rbp-60h]
  __int64 v126; // [rsp+C8h] [rbp-58h]
  __m128i v127; // [rsp+D0h] [rbp-50h]
  __int64 v128; // [rsp+E0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 16);
  if ( v8 && !*(_QWORD *)(v8 + 8) && **(_BYTE **)(v8 + 24) == 67 )
    return 0;
  result = sub_11005E0(a1, (unsigned __int8 *)a2, a3, a4, a5, a6);
  if ( result )
    return result;
  v10 = *(_QWORD *)(a2 - 32);
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(_QWORD *)(v10 + 8);
  v114 = sub_BCB060(v12);
  v110 = sub_BCB060(v11);
  v13 = a1[10].m128i_i64[0];
  v14 = _mm_loadu_si128(a1 + 7);
  v15 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v124[0] = _mm_loadu_si128(a1 + 6);
  v128 = v13;
  v16 = _mm_loadu_si128(a1 + 9);
  v125 = v15;
  v126 = a2;
  v124[1] = v14;
  v127 = v16;
  if ( (unsigned __int8)sub_9AC470(v10, v124, 0) )
  {
    LOWORD(v125) = 257;
    v117 = sub_B51D30(39, v10, v11, (__int64)v124, 0, 0);
    sub_B448D0(v117, 1);
    return (unsigned __int8 *)v117;
  }
  if ( (unsigned __int8)sub_F0C890((__int64)a1, v12, v11) && (unsigned __int8)sub_10FD5E0(v10, v11) )
  {
    v37 = sub_1106750((__int64)a1, (unsigned __int8 *)v10, v11, 1u);
    if ( v110 - v114 < (unsigned int)sub_9AF8B0(
                                       v37,
                                       a1[5].m128i_u64[1],
                                       0,
                                       a1[4].m128i_i64[0],
                                       a2,
                                       a1[5].m128i_i64[0],
                                       1) )
      return sub_F162A0((__int64)a1, a2, v37);
    v68 = sub_AD64C0(v11, (unsigned int)(v110 - v114), 0);
    v69 = a1[2].m128i_i64[0];
    v70 = v68;
    v123 = 257;
    v120[0] = (__int64)"sext";
    v121 = 259;
    v71 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v69 + 80) + 32LL))(
            *(_QWORD *)(v69 + 80),
            25,
            v37,
            v68,
            0,
            0);
    if ( !v71 )
    {
      LOWORD(v125) = 257;
      v71 = sub_B504D0(25, v37, v70, (__int64)v124, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v69 + 88) + 16LL))(
        *(_QWORD *)(v69 + 88),
        v71,
        v120,
        *(_QWORD *)(v69 + 56),
        *(_QWORD *)(v69 + 64));
      v82 = *(_QWORD *)v69;
      v83 = *(_QWORD *)v69 + 16LL * *(unsigned int *)(v69 + 8);
      if ( *(_QWORD *)v69 != v83 )
      {
        do
        {
          v84 = *(_QWORD *)(v82 + 8);
          v85 = *(_DWORD *)v82;
          v82 += 16;
          sub_B99FD0(v71, v85, v84);
        }
        while ( v83 != v82 );
      }
    }
    return (unsigned __int8 *)sub_B504D0(27, v71, v70, (__int64)v122, 0, 0);
  }
  v17 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 > 0x1Cu )
  {
    if ( v17 != 67 )
      goto LABEL_8;
    v63 = *(_QWORD *)(v10 - 32);
    if ( v63 )
    {
      v64 = *(_QWORD *)(v63 + 8);
      v119 = *(_BYTE **)(v10 - 32);
      v102 = v63;
      v107 = sub_BCB060(v64);
      v65 = sub_9AF8B0(v102, a1[5].m128i_u64[1], 0, a1[4].m128i_i64[0], a2, a1[5].m128i_i64[0], 1);
      v66 = (unsigned int)(v107 - v114);
      if ( (unsigned int)v66 < v65 )
      {
        LOWORD(v125) = 257;
        return (unsigned __int8 *)sub_B522D0((__int64)v119, v11, 1, (__int64)v124, 0, 0);
      }
      v67 = *(_QWORD *)(v10 + 16);
      if ( v67 && !*(_QWORD *)(v67 + 8) )
      {
        if ( v11 == *((_QWORD *)v119 + 1) )
        {
          v94 = (_BYTE *)sub_AD64C0(v11, (unsigned int)(v110 - v114), 0);
          v95 = (unsigned int **)a1[2].m128i_i64[0];
          v96 = (__int64)v94;
          LOWORD(v125) = 257;
          v123 = 257;
          v97 = sub_920A70(v95, v119, v94, (__int64)v122, 0, 0);
          return (unsigned __int8 *)sub_B504D0(27, v97, v96, (__int64)v124, 0, 0);
        }
        if ( *v119 == 55 )
        {
          v113 = *((_QWORD *)v119 - 8);
          if ( v113 )
          {
            v72 = *((_QWORD *)v119 - 4);
            if ( !v72 )
              BUG();
            if ( *(_BYTE *)v72 == 17
              || (v108 = v107 - v114, (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v72 + 8) + 8LL) - 17 <= 1)
              && *(_BYTE *)v72 <= 0x15u
              && (v98 = sub_AD7630(v72, 1, v66), (v72 = (__int64)v98) != 0)
              && (v66 = v108, *v98 == 17) )
            {
              if ( *(_DWORD *)(v72 + 32) > 0x40u )
              {
                v100 = *(_DWORD *)(v72 + 32);
                v103 = v66;
                v109 = v72;
                v99 = sub_C444A0(v72 + 24);
                v66 = v103;
                if ( (unsigned int)(v100 - v99) > 0x40 )
                  goto LABEL_53;
                v73 = **(_QWORD **)(v109 + 24);
              }
              else
              {
                v73 = *(_QWORD *)(v72 + 24);
              }
              if ( v66 == v73 )
              {
                v74 = (unsigned int **)a1[2].m128i_i64[0];
                LOWORD(v125) = 257;
                v75 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v113 + 8), v73, 0);
                v76 = sub_920F70(v74, (_BYTE *)v113, v75, (__int64)v124, 0);
                LOWORD(v125) = 257;
                return (unsigned __int8 *)sub_B522D0(v76, v11, 1, (__int64)v124, 0, 0);
              }
            }
          }
        }
      }
LABEL_53:
      v17 = *(_BYTE *)v10;
      if ( *(_BYTE *)v10 > 0x1Cu )
      {
LABEL_8:
        if ( v17 == 82 )
          return sub_1101DF0(a1, v10, a2);
        if ( v17 == 56 )
        {
          v18 = *(_BYTE **)(v10 - 64);
          if ( *v18 == 54 )
          {
            v47 = (_BYTE *)*((_QWORD *)v18 - 8);
            if ( *v47 == 67 )
            {
              v101 = *((_QWORD *)v47 - 4);
              if ( v101 )
              {
                v106 = (unsigned __int8 *)*((_QWORD *)v18 - 4);
                if ( *v106 <= 0x15u )
                {
                  v48 = **(_BYTE **)(v10 - 32);
                  if ( v48 <= 0x15u && v48 != 5 )
                  {
                    v112 = *(unsigned __int8 **)(v10 - 32);
                    if ( !(unsigned __int8)sub_AD6CA0((__int64)v112)
                      && (unsigned __int8)sub_AD8850((unsigned __int64)v106, (unsigned __int64)v112)
                      && v11 == *(_QWORD *)(v101 + 8) )
                    {
                      v49 = (unsigned __int8 *)sub_96F480(0x28u, (__int64)v112, v11, a1[5].m128i_i64[1]);
                      v50 = sub_BCB060(v12);
                      v51 = sub_AD64C0(v11, v50, 0);
                      v52 = (unsigned __int8 *)sub_AD57F0(v51, v49, 0, 0);
                      v53 = sub_BCB060(v11);
                      v54 = sub_AD64C0(v11, v53, 0);
                      v55 = (_BYTE *)sub_AD57F0(v54, v52, 0, 0);
                      v56 = (_BYTE *)sub_AD7180(v55, v106);
                      v57 = sub_AD7180(v56, v112);
                      v58 = a1[2].m128i_i64[0];
                      v59 = v57;
                      v60 = sub_BD5D20(a2);
                      v123 = 261;
                      v122[1] = v61;
                      v122[0] = v60;
                      v62 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v58 + 80) + 32LL))(
                              *(_QWORD *)(v58 + 80),
                              25,
                              v101,
                              v59,
                              0,
                              0);
                      if ( !v62 )
                      {
                        LOWORD(v125) = 257;
                        v62 = sub_B504D0(25, v101, v59, (__int64)v124, 0, 0);
                        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v58 + 88)
                                                                                          + 16LL))(
                          *(_QWORD *)(v58 + 88),
                          v62,
                          v122,
                          *(_QWORD *)(v58 + 56),
                          *(_QWORD *)(v58 + 64));
                        v90 = *(_QWORD *)v58;
                        v91 = *(_QWORD *)v58 + 16LL * *(unsigned int *)(v58 + 8);
                        while ( v91 != v90 )
                        {
                          v92 = *(_QWORD *)(v90 + 8);
                          v93 = *(_DWORD *)v90;
                          v90 += 16;
                          sub_B99FD0(v62, v93, v92);
                        }
                      }
                      LOWORD(v125) = 257;
                      return (unsigned __int8 *)sub_B504D0(27, v62, v59, (__int64)v124, 0, 0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v19 = v114 - 1;
  v124[0].m128i_i64[0] = (__int64)&v119;
  v124[0].m128i_i64[1] = (unsigned int)(v114 - 1);
  v20 = *(_QWORD *)(v10 + 16);
  if ( !v20
    || *(_QWORD *)(v20 + 8)
    || *(_BYTE *)v10 != 56
    || (v21 = *(_BYTE **)(v10 - 64), *v21 != 67)
    || (v22 = (_BYTE *)*((_QWORD *)v21 - 4)) == 0
    || (v119 = v22, !sub_F17ED0(&v124[0].m128i_i64[1], *(_QWORD *)(v10 - 32))) )
  {
LABEL_12:
    if ( (unsigned __int8)sub_10FD760((unsigned __int8 *)v10)
      && sub_B43CB0(a2)
      && (v31 = sub_B43CB0(a2), (unsigned __int8)sub_B2D610(v31, 96))
      && (v32 = sub_B43CB0(a2), v120[0] = sub_B2D7D0(v32, 96), v122[0] = sub_A71ED0(v120), BYTE4(v122[0]))
      && LODWORD(v122[0])
      && (_BitScanReverse(&v33, v122[0]), v19 > 31 - (v33 ^ 0x1F)) )
    {
      v34 = a1[2].m128i_i64[0];
      LOWORD(v125) = 257;
      v35 = sub_AD64C0(v11, 1, 0);
      v36 = sub_B33D80(v34, v35, (__int64)v124);
      return sub_F162A0((__int64)a1, a2, v36);
    }
    else if ( LOBYTE(qword_4F8BA48[8])
           && *(_BYTE *)v10 == 42
           && (v38 = *(_QWORD *)(v10 - 64)) != 0
           && (v39 = *(_BYTE **)(v10 - 32), *v39 == 17) )
    {
      v40 = (__int64 *)a1[2].m128i_i64[0];
      v41 = *(_QWORD *)(a2 + 8);
      LOWORD(v125) = 257;
      v42 = sub_10FF770(v40, 40, v38, v41, (__int64)v124, 0, v122[0], 0);
      v43 = (__int64 *)a1[2].m128i_i64[0];
      v44 = v42;
      v45 = *(_QWORD *)(a2 + 8);
      LOWORD(v125) = 257;
      v46 = sub_10FF770(v43, 40, (__int64)v39, v45, (__int64)v124, 0, v122[0], 0);
      LOWORD(v125) = 257;
      return (unsigned __int8 *)sub_B504D0(13, v44, v46, (__int64)v124, 0, 0);
    }
    else
    {
      return 0;
    }
  }
  v104 = *((_QWORD *)v119 + 1);
  v23 = sub_BCB060(v104);
  v24 = (unsigned int)(v23 - v114);
  v111 = v23;
  v115 = v104;
  v105 = (_BYTE *)sub_AD64C0(v104, v24, 0);
  v25 = sub_AD64C0(v115, (unsigned int)(v111 - 1), 0);
  v26 = (_BYTE *)v25;
  if ( v11 != v115 )
  {
    v27 = *(_QWORD *)(*(_QWORD *)(v10 - 64) + 16LL);
    if ( v27 && !*(_QWORD *)(v27 + 8) )
    {
      v28 = (unsigned int **)a1[2].m128i_i64[0];
      LOWORD(v125) = 257;
      v123 = 257;
      v116 = v26;
      v29 = (_BYTE *)sub_920A70(v28, v119, v105, (__int64)v122, 0, 0);
      v30 = sub_920F70(v28, v29, v116, (__int64)v124, 0);
      LOWORD(v125) = 257;
      return (unsigned __int8 *)sub_B522D0(v30, v11, 1, (__int64)v124, 0, 0);
    }
    goto LABEL_12;
  }
  v77 = a1[2].m128i_i64[0];
  v123 = 257;
  v78 = v119;
  v121 = 257;
  v118 = v25;
  v79 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v77 + 80) + 32LL))(
          *(_QWORD *)(v77 + 80),
          25,
          v119,
          v105,
          0,
          0);
  v80 = v118;
  v81 = v79;
  if ( !v79 )
  {
    LOWORD(v125) = 257;
    v81 = sub_B504D0(25, (__int64)v78, (__int64)v105, (__int64)v124, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v77 + 88) + 16LL))(
      *(_QWORD *)(v77 + 88),
      v81,
      v120,
      *(_QWORD *)(v77 + 56),
      *(_QWORD *)(v77 + 64));
    v86 = *(_QWORD *)v77;
    v80 = v118;
    v87 = *(_QWORD *)v77 + 16LL * *(unsigned int *)(v77 + 8);
    if ( v86 != v87 )
    {
      do
      {
        v88 = *(_QWORD *)(v86 + 8);
        v89 = *(_DWORD *)v86;
        v86 += 16;
        sub_B99FD0(v81, v89, v88);
      }
      while ( v87 != v86 );
      v80 = v118;
    }
  }
  return (unsigned __int8 *)sub_B504D0(27, v81, v80, (__int64)v122, 0, 0);
}
