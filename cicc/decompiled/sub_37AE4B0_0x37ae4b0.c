// Function: sub_37AE4B0
// Address: 0x37ae4b0
//
__m128i *__fastcall sub_37AE4B0(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned __int64 *v5; // rax
  __int64 v6; // r9
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  __int16 v12; // dx
  __int16 *v13; // rax
  unsigned __int16 v14; // si
  __int64 v15; // r8
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int16 v19; // cx
  __m128i v20; // rax
  _QWORD *v21; // rdx
  __int8 v22; // al
  __m128i v23; // rax
  unsigned int v24; // eax
  __int64 v25; // r8
  unsigned __int16 v26; // dx
  bool v27; // al
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int16 v33; // dx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  unsigned int v38; // ebx
  int v39; // eax
  int v40; // r9d
  __int64 v41; // rdx
  unsigned int v42; // r11d
  __int16 v43; // r10
  _QWORD *v44; // rdi
  __int64 v45; // r14
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  unsigned int v49; // eax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rsi
  int v53; // edx
  int v54; // r15d
  __int16 v55; // r10
  unsigned int v56; // r11d
  unsigned __int64 v57; // r14
  _QWORD *v58; // rcx
  __int64 *v59; // rdx
  unsigned __int64 v60; // rcx
  _QWORD *v61; // rdi
  int v62; // r9d
  unsigned __int64 v63; // rdi
  __m128i *v64; // r12
  __int128 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdi
  __int64 v69; // rcx
  __int64 v70; // r8
  unsigned int v71; // edx
  unsigned __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  unsigned int v80; // eax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdx
  unsigned __int16 v84; // ax
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rdx
  bool v89; // al
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r8
  __int64 v93; // r9
  _QWORD *v94; // rax
  unsigned int v95; // r11d
  __int64 v96; // r9
  __int64 v97; // rdx
  __m128i v98; // rax
  __int8 v99; // al
  unsigned int v100; // eax
  __int64 v101; // r13
  unsigned int v102; // ebx
  __int64 v103; // rax
  __int64 v104; // rax
  _QWORD *v105; // r13
  __int64 v106; // rdx
  __int128 v107; // rax
  __int64 v108; // r9
  __int128 v109; // [rsp-10h] [rbp-250h]
  __int128 v110; // [rsp-10h] [rbp-250h]
  __int64 v111; // [rsp-8h] [rbp-248h]
  __int16 v112; // [rsp+Ah] [rbp-236h]
  __int16 v113; // [rsp+14h] [rbp-22Ch]
  __int16 v114; // [rsp+1Ah] [rbp-226h]
  unsigned int v115; // [rsp+20h] [rbp-220h]
  __int16 v116; // [rsp+22h] [rbp-21Eh]
  unsigned int v117; // [rsp+28h] [rbp-218h]
  __int64 v118; // [rsp+28h] [rbp-218h]
  unsigned int v119; // [rsp+30h] [rbp-210h]
  unsigned __int16 v120; // [rsp+30h] [rbp-210h]
  unsigned int v121; // [rsp+40h] [rbp-200h]
  __int16 v122; // [rsp+40h] [rbp-200h]
  __int128 v123; // [rsp+40h] [rbp-200h]
  unsigned __int16 v124; // [rsp+40h] [rbp-200h]
  __int16 v125; // [rsp+40h] [rbp-200h]
  unsigned int v126; // [rsp+50h] [rbp-1F0h]
  __int64 v127; // [rsp+50h] [rbp-1F0h]
  unsigned __int64 v128; // [rsp+58h] [rbp-1E8h]
  __m128i v129; // [rsp+90h] [rbp-1B0h] BYREF
  unsigned int v130; // [rsp+A0h] [rbp-1A0h] BYREF
  unsigned __int64 v131; // [rsp+A8h] [rbp-198h]
  __int64 v132; // [rsp+B0h] [rbp-190h] BYREF
  int v133; // [rsp+B8h] [rbp-188h]
  unsigned int v134; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v135; // [rsp+C8h] [rbp-178h]
  __m128i v136; // [rsp+D0h] [rbp-170h] BYREF
  __m128i v137; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v138; // [rsp+F0h] [rbp-150h]
  __int64 v139; // [rsp+F8h] [rbp-148h]
  _QWORD *v140; // [rsp+100h] [rbp-140h] BYREF
  __int64 v141; // [rsp+108h] [rbp-138h]
  _QWORD v142[38]; // [rsp+110h] [rbp-130h] BYREF

  v5 = *(unsigned __int64 **)(a2 + 40);
  v6 = *a1;
  v7 = *v5;
  v8 = *v5;
  v9 = v5[1];
  v10 = *((unsigned int *)v5 + 2);
  v128 = v7;
  v126 = v10;
  v11 = *(_QWORD *)(v7 + 48) + 16 * v10;
  v12 = *(_WORD *)v11;
  v129.m128i_i64[1] = *(_QWORD *)(v11 + 8);
  v13 = *(__int16 **)(a2 + 48);
  v129.m128i_i16[0] = v12;
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v16 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v140, v6, *(_QWORD *)(a1[1] + 64), v14, v15);
    LOWORD(v130) = v141;
    v131 = v142[0];
  }
  else
  {
    v130 = v16(v6, *(_QWORD *)(a1[1] + 64), v14, v15);
    v131 = v79;
  }
  v17 = *(_QWORD *)(a2 + 80);
  v132 = v17;
  if ( v17 )
    sub_B96E90((__int64)&v132, v17, 1);
  v18 = *a1;
  v133 = *(_DWORD *)(a2 + 72);
  sub_2FE6CC0((__int64)&v140, v18, *(_QWORD *)(a1[1] + 64), v129.m128i_u16[0], v129.m128i_i64[1]);
  if ( (_BYTE)v140 != 7 )
  {
    if ( (_BYTE)v140 == 10 )
      sub_C64ED0("Scalarization of scalable vectors is not supported.", 1u);
    if ( (_BYTE)v140 != 1 )
      goto LABEL_8;
    if ( v129.m128i_i16[0] )
    {
      v19 = v130;
      if ( (unsigned __int16)(v129.m128i_i16[0] - 17) <= 0xD3u )
        goto LABEL_9;
    }
    else if ( sub_30070B0((__int64)&v129) )
    {
LABEL_8:
      v19 = v130;
      goto LABEL_9;
    }
    *(_QWORD *)&v66 = sub_37AE0F0((__int64)a1, v8, v9);
    v18 = DWORD2(v66);
    v123 = v66;
    v128 = v66;
    *(_QWORD *)&v66 = *(_QWORD *)(v66 + 48) + 16LL * DWORD2(v66);
    WORD4(v66) = *(_WORD *)v66;
    v67 = *(_QWORD *)(v66 + 8);
    v126 = v18;
    v136.m128i_i16[0] = WORD4(v66);
    v136.m128i_i64[1] = v67;
    v137 = _mm_loadu_si128(&v136);
    if ( WORD4(v66) == (_WORD)v130 && ((_WORD)v130 || v67 == v131)
      || (v120 = v130,
          v140 = (_QWORD *)sub_2D5B750((unsigned __int16 *)&v137),
          v141 = v86,
          v87 = sub_2D5B750((unsigned __int16 *)&v130),
          v19 = v120,
          v138 = v87,
          v139 = v88,
          (_QWORD *)v87 == v140)
      && (_BYTE)v139 == (_BYTE)v141 )
    {
      if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
      {
        v138 = sub_2D5B750((unsigned __int16 *)&v129);
        v139 = v97;
        v98.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v136);
        v137 = v98;
        v98.m128i_i64[1] = v98.m128i_i64[0];
        v99 = v137.m128i_i8[8];
        v140 = (_QWORD *)(v98.m128i_i64[1] - v138);
        if ( v138 )
          v99 = v139;
        LOBYTE(v141) = v99;
        v100 = sub_CA1930(&v140);
        v101 = *a1;
        v102 = v100;
        v103 = sub_2E79000(*(__int64 **)(a1[1] + 40));
        v104 = sub_2FE6750(v101, v136.m128i_u32[0], v136.m128i_i64[1], v103);
        v105 = (_QWORD *)a1[1];
        *(_QWORD *)&v107 = sub_3400BD0((__int64)v105, v102, (__int64)&v132, v104, v106, 0, a3, 0);
        sub_3406EB0(v105, 0xBEu, (__int64)&v132, v136.m128i_u32[0], v136.m128i_i64[1], v108, v123, v107);
      }
      v68 = a1[1];
      v69 = v130;
      v70 = v131;
      goto LABEL_53;
    }
    v129 = _mm_loadu_si128(&v136);
LABEL_9:
    if ( v19 )
    {
      if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
        goto LABEL_87;
      v78 = 16LL * (v19 - 1);
      v21 = *(_QWORD **)&byte_444C4A0[v78];
      v22 = byte_444C4A0[v78 + 8];
    }
    else
    {
      v20.m128i_i64[0] = sub_3007260((__int64)&v130);
      v137 = v20;
      v21 = (_QWORD *)v20.m128i_i64[0];
      v22 = v137.m128i_i8[8];
    }
    v140 = v21;
    LOBYTE(v141) = v22;
    v117 = sub_CA1930(&v140);
    v121 = v117;
    if ( v129.m128i_i16[0] )
    {
      if ( v129.m128i_i16[0] == 1 || (unsigned __int16)(v129.m128i_i16[0] - 504) <= 7u )
        goto LABEL_87;
      v23.m128i_i64[1] = 16LL * (v129.m128i_u16[0] - 1);
      v23.m128i_i64[0] = *(_QWORD *)&byte_444C4A0[v23.m128i_i64[1]];
      v23.m128i_i8[8] = byte_444C4A0[v23.m128i_i64[1] + 8];
    }
    else
    {
      v23.m128i_i64[0] = sub_3007260((__int64)&v129);
      v136 = v23;
    }
    LOBYTE(v141) = v23.m128i_i8[8];
    v140 = (_QWORD *)v23.m128i_i64[0];
    v24 = sub_CA1930(&v140);
    v26 = v129.m128i_i16[0];
    v119 = v24;
    if ( v129.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v129.m128i_i16[0] - 17) <= 0xD3u )
      {
        v26 = word_4456580[v129.m128i_u16[0] - 1];
        v29 = 0;
        goto LABEL_16;
      }
    }
    else
    {
      v27 = sub_30070B0((__int64)&v129);
      v26 = 0;
      if ( v27 )
      {
        v84 = sub_3009970((__int64)&v129, v18, 0, v28, v25);
        v25 = v85;
        v26 = v84;
        v29 = v25;
        goto LABEL_16;
      }
    }
    v29 = v129.m128i_i64[1];
LABEL_16:
    LOWORD(v140) = v26;
    v141 = v29;
    if ( !v26 )
    {
      v138 = sub_3007260((__int64)&v140);
      v30 = v138;
      v139 = v31;
LABEL_18:
      v115 = v117 / (unsigned int)v30;
      if ( v117 % (unsigned int)v30 || v129.m128i_i16[0] == 261 )
        goto LABEL_40;
      if ( v129.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v129.m128i_i16[0] - 17) > 0xD3u )
        {
LABEL_22:
          v32 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
          v33 = *(_WORD *)v32;
          v34 = *(_QWORD *)(v32 + 8);
          LOWORD(v134) = v33;
          v135 = v34;
          v35 = sub_2D5B750((unsigned __int16 *)&v134);
          v141 = v36;
          v140 = (_QWORD *)v35;
          v37 = v117 / (unsigned __int64)sub_CA1930(&v140);
          v38 = v37;
LABEL_23:
          v39 = sub_327FCF0(*(__int64 **)(a1[1] + 64), v134, v135, v37, 0);
          v118 = v41;
          HIWORD(v42) = HIWORD(v39);
          v43 = v39;
          if ( (_WORD)v39 && *(_QWORD *)(*a1 + 8LL * (unsigned __int16)v39 + 112) )
          {
            if ( v129.m128i_i16[0] )
            {
              if ( (unsigned __int16)(v129.m128i_i16[0] - 17) <= 0xD3u )
              {
LABEL_27:
                v44 = (_QWORD *)a1[1];
                if ( v121 % v119 )
                {
                  v141 = 0x1000000000LL;
                  v114 = HIWORD(v42);
                  v122 = v43;
                  v140 = v142;
                  sub_3408690(v44, v128, v126 | v9 & 0xFFFFFFFF00000000LL, (unsigned __int16 *)&v140, 0, 0, a3, 0, 0);
                  v45 = a1[1];
                  v49 = sub_3281170(&v129, v128, v46, v47, v48);
                  v51 = sub_3288990(v45, v49, v50);
                  v52 = (unsigned int)v141;
                  v54 = v53;
                  v55 = v122;
                  HIWORD(v56) = v114;
                  v57 = v115 - (unsigned __int64)(unsigned int)v141;
                  if ( v115 > (unsigned __int64)HIDWORD(v141) )
                  {
                    v127 = v51;
                    sub_C8D5F0((__int64)&v140, v142, v115, 0x10u, v115, v111);
                    v52 = (unsigned int)v141;
                    HIWORD(v56) = v114;
                    v51 = v127;
                    v55 = v122;
                  }
                  v58 = v140;
                  v59 = &v140[2 * v52];
                  if ( v57 )
                  {
                    v60 = v57;
                    do
                    {
                      if ( v59 )
                      {
                        *v59 = v51;
                        *((_DWORD *)v59 + 2) = v54;
                      }
                      v59 += 2;
                      --v60;
                    }
                    while ( v60 );
                    v58 = v140;
                    v52 = (unsigned int)v141;
                  }
                  v61 = (_QWORD *)a1[1];
                  LOWORD(v56) = v55;
                  LODWORD(v141) = v57 + v52;
                  *((_QWORD *)&v109 + 1) = (unsigned int)(v57 + v52);
                  *(_QWORD *)&v109 = v58;
                  sub_33FC220(v61, 156, (__int64)&v132, v56, v118, v57 + v52, v109);
                  v63 = (unsigned __int64)v140;
                  if ( v140 == v142 )
                    goto LABEL_80;
                  goto LABEL_79;
                }
                v116 = HIWORD(v42);
                v125 = v43;
                v90 = sub_3288990((__int64)v44, v129.m128i_u32[0], v129.m128i_i64[1]);
                v140 = v142;
                v141 = 0x1000000000LL;
                sub_32982C0((__int64)&v140, v38, v90, v91, v92, v93);
                v94 = v140;
                HIWORD(v95) = v116;
                *v140 = v128;
                LOWORD(v95) = v125;
                *((_DWORD *)v94 + 2) = v126;
                *((_QWORD *)&v110 + 1) = (unsigned int)v141;
                *(_QWORD *)&v110 = v140;
                sub_33FC220((_QWORD *)a1[1], 159, (__int64)&v132, v95, v118, v96, v110);
                v63 = (unsigned __int64)v140;
                if ( v140 != v142 )
LABEL_79:
                  _libc_free(v63);
LABEL_80:
                v68 = a1[1];
                v69 = v130;
                v70 = v131;
                goto LABEL_53;
              }
            }
            else
            {
              v112 = HIWORD(v39);
              v113 = v39;
              v89 = sub_30070B0((__int64)&v129);
              v43 = v113;
              HIWORD(v42) = v112;
              if ( v89 )
                goto LABEL_27;
            }
            LOWORD(v42) = v43;
            sub_33FAF80(a1[1], 167, (__int64)&v132, v42, v118, v40, a3);
            goto LABEL_80;
          }
LABEL_40:
          v64 = sub_375AC00((__int64)a1, v128, v9 & 0xFFFFFFFF00000000LL | v126, v130, v131);
          goto LABEL_41;
        }
      }
      else if ( !sub_30070B0((__int64)&v129) )
      {
        goto LABEL_22;
      }
      v38 = v117 / v119;
      v80 = sub_3281170(&v129, v119, v117 % v119, v30, v25);
      v135 = v81;
      v134 = v80;
      v82 = sub_2D5B750((unsigned __int16 *)&v134);
      v141 = v83;
      v140 = (_QWORD *)v82;
      v37 = v117 / (unsigned __int64)sub_CA1930(&v140);
      goto LABEL_23;
    }
    if ( v26 != 1 && (unsigned __int16)(v26 - 504) > 7u )
    {
      v30 = *(_QWORD *)&byte_444C4A0[16 * v26 - 16];
      goto LABEL_18;
    }
LABEL_87:
    BUG();
  }
  v18 = sub_379AB60((__int64)a1, v8, v9);
  v128 = v18;
  v126 = v71;
  v72 = v71 | v9 & 0xFFFFFFFF00000000LL;
  v73 = *(_QWORD *)(v18 + 48) + 16LL * v71;
  v9 = v72;
  LOWORD(v72) = *(_WORD *)v73;
  v74 = *(_QWORD *)(v73 + 8);
  v129.m128i_i16[0] = v72;
  v129.m128i_i64[1] = v74;
  a3 = _mm_loadu_si128(&v129);
  v137 = a3;
  if ( (_WORD)v72 != (_WORD)v130 || !(_WORD)v130 && v74 != v131 )
  {
    v124 = v130;
    v140 = (_QWORD *)sub_2D5B750((unsigned __int16 *)&v137);
    v141 = v75;
    v76 = sub_2D5B750((unsigned __int16 *)&v130);
    v19 = v124;
    v138 = v76;
    v139 = v77;
    if ( (_QWORD *)v76 != v140 || (_BYTE)v139 != (_BYTE)v141 )
      goto LABEL_9;
  }
  v68 = a1[1];
  v69 = v130;
  v70 = v131;
LABEL_53:
  v64 = (__m128i *)sub_33FAF80(v68, 234, (__int64)&v132, v69, v70, v62, a3);
LABEL_41:
  if ( v132 )
    sub_B91220((__int64)&v132, v132);
  return v64;
}
