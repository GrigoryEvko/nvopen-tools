// Function: sub_37AB210
// Address: 0x37ab210
//
unsigned __int8 *__fastcall sub_37AB210(_QWORD *a1, __int64 a2, __m128i a3)
{
  unsigned __int8 *v3; // rcx
  _QWORD *v4; // r12
  __int64 v5; // rsi
  int v6; // eax
  unsigned __int8 *v7; // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  unsigned __int8 *v10; // r14
  unsigned __int64 v11; // rdx
  __int64 v12; // rbx
  unsigned __int64 v13; // r15
  __int16 *v14; // rax
  __int16 v15; // si
  __int64 v16; // rdx
  unsigned __int16 *v17; // rdx
  unsigned int *v18; // rax
  unsigned __int16 *v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned int *v22; // rsi
  int v23; // eax
  int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // rdx
  unsigned __int32 v27; // r11d
  int v28; // eax
  __int64 v29; // rax
  int v30; // r9d
  unsigned __int32 v31; // r11d
  unsigned __int16 v32; // dx
  __int64 v33; // rax
  unsigned __int64 v34; // rbx
  bool v35; // al
  char v36; // cl
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // rdi
  int v40; // eax
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rcx
  unsigned int v44; // esi
  __m128i v45; // rax
  unsigned int *v46; // r10
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // rdi
  __int64 (*v50)(); // rdx
  unsigned __int16 v51; // ax
  unsigned __int8 *v52; // rax
  _QWORD *v53; // rdi
  __int64 v54; // rdx
  __m128i v55; // xmm0
  unsigned __int8 *v56; // r14
  int v57; // r10d
  unsigned __int32 v58; // eax
  unsigned __int32 i; // ecx
  int v60; // edx
  __int64 *v61; // rax
  unsigned __int16 v62; // ax
  __int64 v63; // r9
  __int64 v64; // r8
  int v65; // r10d
  unsigned __int32 v66; // r11d
  _QWORD *v67; // rdi
  __int128 v68; // rax
  int v69; // r10d
  unsigned __int32 v70; // r11d
  unsigned __int32 v71; // r12d
  __int128 v72; // rax
  unsigned int v73; // edx
  bool v74; // al
  __int64 v75; // r12
  __int64 v76; // rsi
  __int128 v77; // rax
  __int64 v78; // r9
  unsigned int v79; // edx
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rdi
  __int64 v84; // rsi
  __int64 v85; // rdx
  __int128 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rdx
  __int128 v89; // [rsp-30h] [rbp-180h]
  __int128 v90; // [rsp-30h] [rbp-180h]
  __int64 *v91; // [rsp+0h] [rbp-150h]
  __int16 v92; // [rsp+2h] [rbp-14Eh]
  __int16 v93; // [rsp+Ch] [rbp-144h]
  unsigned __int32 v94; // [rsp+Ch] [rbp-144h]
  int v95; // [rsp+Ch] [rbp-144h]
  int v96; // [rsp+10h] [rbp-140h]
  int v97; // [rsp+14h] [rbp-13Ch]
  unsigned __int32 v98; // [rsp+18h] [rbp-138h]
  int v99; // [rsp+18h] [rbp-138h]
  int v100; // [rsp+18h] [rbp-138h]
  int v101; // [rsp+18h] [rbp-138h]
  unsigned __int32 v102; // [rsp+18h] [rbp-138h]
  unsigned __int32 v103; // [rsp+18h] [rbp-138h]
  __int64 v104; // [rsp+20h] [rbp-130h]
  __int128 v105; // [rsp+20h] [rbp-130h]
  __int128 v106; // [rsp+30h] [rbp-120h]
  _QWORD *v107; // [rsp+30h] [rbp-120h]
  int v108; // [rsp+30h] [rbp-120h]
  __m128i v109; // [rsp+40h] [rbp-110h] BYREF
  unsigned int *v110; // [rsp+50h] [rbp-100h]
  unsigned __int8 *v111; // [rsp+58h] [rbp-F8h]
  unsigned __int8 *v112; // [rsp+60h] [rbp-F0h]
  __int64 v113; // [rsp+68h] [rbp-E8h]
  __int64 v114; // [rsp+70h] [rbp-E0h]
  __int64 v115; // [rsp+78h] [rbp-D8h]
  __int64 v116; // [rsp+80h] [rbp-D0h]
  __int64 v117; // [rsp+88h] [rbp-C8h]
  __int64 v118; // [rsp+90h] [rbp-C0h]
  unsigned __int64 v119; // [rsp+98h] [rbp-B8h]
  __int64 v120; // [rsp+A0h] [rbp-B0h] BYREF
  int v121; // [rsp+A8h] [rbp-A8h]
  unsigned int v122; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v123; // [rsp+B8h] [rbp-98h]
  unsigned __int16 v124; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v125; // [rsp+C8h] [rbp-88h]
  unsigned int v126; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v127; // [rsp+D8h] [rbp-78h]
  unsigned __int8 *v128; // [rsp+E0h] [rbp-70h] BYREF
  unsigned __int64 v129; // [rsp+E8h] [rbp-68h]
  unsigned __int8 *v130; // [rsp+F0h] [rbp-60h]
  unsigned __int64 v131; // [rsp+F8h] [rbp-58h]
  __m128i v132; // [rsp+100h] [rbp-50h]
  unsigned __int8 *v133; // [rsp+110h] [rbp-40h]
  __int64 v134; // [rsp+118h] [rbp-38h]

  v3 = (unsigned __int8 *)a2;
  v4 = a1;
  v5 = *(_QWORD *)(a2 + 80);
  v120 = v5;
  if ( v5 )
  {
    v111 = v3;
    sub_B96E90((__int64)&v120, v5, 1);
    v3 = v111;
  }
  v6 = *((_DWORD *)v3 + 18);
  v109.m128i_i64[0] = (__int64)v3;
  v121 = v6;
  v7 = (unsigned __int8 *)sub_379AB60((__int64)a1, **((_QWORD **)v3 + 5), *(_QWORD *)(*((_QWORD *)v3 + 5) + 8LL));
  v9 = v109.m128i_i64[0];
  v10 = v7;
  v111 = v7;
  v12 = (unsigned int)v11;
  v13 = v11;
  v14 = *(__int16 **)(v109.m128i_i64[0] + 48);
  v15 = *v14;
  v123 = *((_QWORD *)v14 + 1);
  v18 = *(unsigned int **)(v109.m128i_i64[0] + 40);
  v93 = v15;
  v16 = v18[2];
  LOWORD(v122) = v15;
  v17 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v18 + 48LL) + 16 * v16);
  LODWORD(v18) = *v17;
  v125 = *((_QWORD *)v17 + 1);
  v19 = (unsigned __int16 *)(*((_QWORD *)v10 + 6) + 16 * v12);
  v124 = (unsigned __int16)v18;
  v20 = *v19;
  v21 = *((_QWORD *)v19 + 1);
  LOWORD(v126) = v20;
  v127 = v21;
  if ( (_WORD)v18 )
  {
    v104 = 0;
    LOWORD(v18) = word_4456580[(int)v18 - 1];
  }
  else
  {
    v18 = (unsigned int *)sub_3009970((__int64)&v124, v20, v21, v109.m128i_i64[0], v8);
    v9 = v109.m128i_i64[0];
    v110 = v18;
    v104 = v82;
  }
  v22 = v110;
  LOWORD(v22) = (_WORD)v18;
  v23 = *(_DWORD *)(v9 + 28);
  v24 = *(_DWORD *)(v9 + 24);
  v110 = v22;
  v96 = v24;
  v97 = v23;
  v25 = sub_33CB000(v24);
  *(_QWORD *)&v106 = sub_3401F50(a1[1], v25, (__int64)&v120, (unsigned int)v22, v104, v97, a3);
  *((_QWORD *)&v106 + 1) = v26;
  if ( v124 )
  {
    v27 = word_4456340[v124 - 1];
    v28 = (unsigned __int16)v126;
    if ( (_WORD)v126 )
    {
LABEL_7:
      v109.m128i_i32[0] = word_4456340[v28 - 1];
      goto LABEL_8;
    }
  }
  else
  {
    v117 = sub_3007240((__int64)&v124);
    v27 = v117;
    v28 = (unsigned __int16)v126;
    if ( (_WORD)v126 )
      goto LABEL_7;
  }
  v102 = v27;
  v81 = sub_3007240((__int64)&v126);
  v27 = v102;
  v116 = v81;
  v109.m128i_i32[0] = v81;
LABEL_8:
  v98 = v27;
  v29 = sub_33CB7C0(v96);
  v31 = v98;
  v114 = v29;
  if ( !BYTE4(v29) )
  {
    v32 = v126;
    if ( (_WORD)v126 )
      goto LABEL_27;
    goto LABEL_43;
  }
  v32 = v126;
  v33 = *a1;
  if ( (_WORD)v126 == 1 )
    goto LABEL_10;
  if ( !(_WORD)v126 )
  {
LABEL_43:
    v74 = sub_3007100((__int64)&v126);
    v31 = v98;
    if ( v74 )
    {
LABEL_28:
      v57 = v109.m128i_i32[0];
      if ( v31 )
      {
        v57 = v31;
        if ( v109.m128i_i32[0] )
        {
          v58 = v109.m128i_i32[0];
          for ( i = v31 % v109.m128i_i32[0]; i; i = v60 )
          {
            v60 = v58 % i;
            v58 = i;
          }
          v57 = v58;
        }
      }
      v94 = v31;
      v99 = v57;
      v61 = *(__int64 **)(a1[1] + 64LL);
      LODWORD(v128) = v57;
      BYTE4(v128) = 1;
      v91 = v61;
      v62 = sub_2D43AD0((__int16)v110, v57);
      v64 = 0;
      v65 = v99;
      v66 = v94;
      if ( !v62 )
      {
        v95 = v99;
        v103 = v66;
        v62 = sub_3009450(v91, (unsigned int)v110, v104, (__int64)v128, 0, v63);
        v65 = v95;
        v66 = v103;
        v64 = v87;
      }
      v67 = (_QWORD *)a1[1];
      if ( *(_DWORD *)(v106 + 24) == 51 )
      {
        v108 = v65;
        LODWORD(v110) = v66;
        v128 = 0;
        LODWORD(v129) = 0;
        *(_QWORD *)&v86 = sub_33F17F0(v67, 51, (__int64)&v128, v62, v64);
        v70 = (unsigned int)v110;
        v105 = v86;
        v69 = v108;
        if ( v128 )
        {
          sub_B91220((__int64)&v128, (__int64)v128);
          v70 = (unsigned int)v110;
          v69 = v108;
        }
      }
      else
      {
        v100 = v65;
        LODWORD(v110) = v66;
        *(_QWORD *)&v68 = sub_33FAF80((__int64)v67, 168, (__int64)&v120, v62, v64, v63, a3);
        v69 = v100;
        v70 = (unsigned int)v110;
        v105 = v68;
      }
      if ( v70 < v109.m128i_i32[0] )
      {
        v101 = v69;
        v107 = v4;
        v71 = v70;
        do
        {
          v110 = (unsigned int *)v107[1];
          *(_QWORD *)&v72 = sub_3400EE0((__int64)v110, v71, (__int64)&v120, 0, a3);
          v13 = v12 | v13 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v89 + 1) = v13;
          *(_QWORD *)&v89 = v111;
          v71 += v101;
          v111 = (unsigned __int8 *)sub_340F900(
                                      v110,
                                      0xA0u,
                                      (__int64)&v120,
                                      v126,
                                      v127,
                                      *((__int64 *)&v72 + 1),
                                      v89,
                                      v105,
                                      v72);
          v12 = v73;
        }
        while ( v71 < v109.m128i_i32[0] );
        v4 = v107;
      }
LABEL_48:
      v56 = sub_33FA050(
              v4[1],
              (unsigned int)v96,
              (__int64)&v120,
              v122,
              v123,
              v97,
              v111,
              v12 | v13 & 0xFFFFFFFF00000000LL);
      goto LABEL_49;
    }
LABEL_44:
    if ( v31 < v109.m128i_i32[0] )
    {
      v75 = v31;
      do
      {
        v76 = v75++;
        v110 = (unsigned int *)a1[1];
        *(_QWORD *)&v77 = sub_3400EE0((__int64)v110, v76, (__int64)&v120, 0, a3);
        v13 = v12 | v13 & 0xFFFFFFFF00000000LL;
        *((_QWORD *)&v90 + 1) = v13;
        *(_QWORD *)&v90 = v111;
        v111 = (unsigned __int8 *)sub_340F900(v110, 0x9Du, (__int64)&v120, v126, v127, v78, v90, v106, v77);
        v12 = v79;
      }
      while ( v109.m128i_i32[0] > (unsigned int)v75 );
      v4 = a1;
    }
    goto LABEL_48;
  }
  if ( !*(_QWORD *)(v33 + 8LL * (unsigned __int16)v126 + 112) )
    goto LABEL_27;
LABEL_10:
  if ( (unsigned int)v114 <= 0x1F3
    && (*(_BYTE *)((unsigned int)v114 + 500LL * (unsigned __int16)v126 + v33 + 6414) & 0xFB) != 0 )
  {
LABEL_27:
    if ( (unsigned __int16)(v32 - 176) <= 0x34u )
      goto LABEL_28;
    goto LABEL_44;
  }
  v34 = *((_QWORD *)&v106 + 1);
  v111 = (unsigned __int8 *)v106;
  if ( !v93 )
  {
    v109.m128i_i32[0] = (unsigned __int16)v126;
    v35 = sub_3007070((__int64)&v122);
    v32 = v109.m128i_i16[0];
    if ( !v35 )
      goto LABEL_14;
LABEL_58:
    v83 = a1[1];
    if ( v96 > 388 )
    {
      if ( (unsigned int)(v96 - 389) <= 1 )
      {
        v84 = 214;
        goto LABEL_62;
      }
    }
    else
    {
      if ( v96 > 386 )
      {
        v84 = 213;
        goto LABEL_62;
      }
      if ( (unsigned int)(v96 - 382) <= 4 )
      {
        v84 = 215;
LABEL_62:
        v112 = sub_33FAF80(v83, v84, (__int64)&v120, v122, v123, v30, a3);
        v113 = v85;
        v32 = v126;
        v111 = v112;
        v34 = (unsigned int)v113 | *((_QWORD *)&v106 + 1) & 0xFFFFFFFF00000000LL;
        goto LABEL_14;
      }
    }
    BUG();
  }
  if ( (unsigned __int16)(v93 - 2) <= 7u
    || (unsigned __int16)(v93 - 17) <= 0x6Cu
    || (unsigned __int16)(v93 - 176) <= 0x1Fu )
  {
    goto LABEL_58;
  }
LABEL_14:
  if ( v32 )
  {
    v36 = (unsigned __int16)(v32 - 176) <= 0x34u;
    LODWORD(v37) = word_4456340[v32 - 1];
    LOBYTE(v38) = v36;
  }
  else
  {
    v37 = sub_3007240((__int64)&v126);
    v38 = HIDWORD(v37);
    v119 = v37;
    v36 = BYTE4(v37);
  }
  v39 = *(_QWORD *)(v4[1] + 64LL);
  LODWORD(v128) = v37;
  BYTE4(v128) = v38;
  v109.m128i_i64[0] = v39;
  if ( v36 )
    LOWORD(v40) = sub_2D43AD0(2, v37);
  else
    LOWORD(v40) = sub_2D43050(2, v37);
  v43 = 0;
  if ( !(_WORD)v40 )
  {
    v40 = sub_3009450((__int64 *)v109.m128i_i64[0], 2, 0, (__int64)v128, v41, v42);
    v92 = HIWORD(v40);
    v43 = v88;
  }
  HIWORD(v44) = v92;
  LOWORD(v44) = v40;
  v45.m128i_i64[0] = (__int64)sub_34015B0(v4[1], (__int64)&v120, v44, v43, 0, 0, a3);
  v46 = (unsigned int *)v4[1];
  v109 = v45;
  if ( v124 )
  {
    LOBYTE(v47) = (unsigned __int16)(v124 - 176) <= 0x34u;
    LODWORD(v48) = word_4456340[v124 - 1];
  }
  else
  {
    v110 = v46;
    v48 = sub_3007240((__int64)&v124);
    v46 = v110;
    v118 = v48;
    v47 = HIDWORD(v48);
  }
  BYTE4(v118) = v47;
  v49 = *v4;
  LODWORD(v118) = v48;
  v115 = v118;
  v50 = *(__int64 (**)())(*(_QWORD *)v49 + 80LL);
  v51 = 7;
  if ( v50 != sub_2FE2E20 )
  {
    v110 = v46;
    v51 = ((__int64 (__fastcall *)(__int64))v50)(v49);
    v46 = v110;
  }
  v52 = sub_3401C20((__int64)v46, (__int64)&v120, v51, 0, v115, a3);
  v130 = v10;
  v53 = (_QWORD *)v4[1];
  v133 = v52;
  v134 = v54;
  v55 = _mm_load_si128(&v109);
  v128 = v111;
  v131 = v13;
  v129 = v34;
  v132 = v55;
  v56 = sub_33FBA10(v53, (unsigned int)v114, (__int64)&v120, v122, v123, v97, (__int64)&v128, 4);
LABEL_49:
  if ( v120 )
    sub_B91220((__int64)&v120, v120);
  return v56;
}
