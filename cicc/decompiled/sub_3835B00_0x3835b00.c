// Function: sub_3835B00
// Address: 0x3835b00
//
__int64 __fastcall sub_3835B00(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // rbx
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // ecx
  __int64 v13; // rbx
  __int64 v14; // rsi
  unsigned __int16 *v15; // rdx
  __int64 v16; // r9
  _QWORD *v17; // rdi
  __int64 v18; // r14
  __int64 v19; // r15
  __int64 v20; // r12
  int *v22; // rbx
  char v23; // si
  __int64 v24; // rax
  _QWORD *v25; // r8
  int v26; // ecx
  unsigned int v27; // edi
  _QWORD *v28; // rax
  int v29; // r9d
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r15
  __int16 v34; // dx
  __int64 v35; // rax
  unsigned __int16 *v36; // rax
  unsigned int v37; // ecx
  __int64 v38; // rbx
  unsigned int v39; // r13d
  int v40; // eax
  int v41; // r9d
  __int64 v42; // r8
  unsigned int v43; // ebx
  __int64 v44; // rdx
  __int64 v45; // r8
  unsigned __int16 v46; // ax
  unsigned int v47; // r13d
  int v48; // eax
  int v49; // r9d
  __int64 v50; // r8
  unsigned int v51; // edi
  unsigned __int8 *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r15
  unsigned __int8 *v55; // r14
  __int128 v56; // rax
  __int64 v57; // r9
  unsigned __int16 *v58; // r9
  int v59; // eax
  __int64 v60; // rdx
  unsigned int v61; // ebx
  unsigned __int64 v62; // r14
  char v63; // r15
  unsigned int v64; // ebx
  unsigned __int16 v65; // cx
  int v66; // esi
  unsigned int v67; // ebx
  __int64 *v68; // rax
  int v69; // eax
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // r14
  unsigned int v73; // ebx
  __int64 v74; // r9
  _QWORD *v75; // rdi
  int v76; // edx
  __int64 v77; // r9
  int v78; // edx
  __int64 v79; // r9
  unsigned int v80; // edx
  __int64 v81; // rdx
  bool v82; // al
  __int64 v83; // rdx
  __int64 v84; // r8
  unsigned __int16 v85; // ax
  __int64 v86; // rdx
  bool v87; // al
  __int64 v88; // rdx
  __int64 v89; // r8
  __int64 v90; // rdx
  unsigned __int8 *v91; // rax
  __int64 v92; // rdi
  int v93; // edx
  int v94; // r9d
  __int64 v95; // rdx
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rdx
  __int64 v99; // rax
  int i; // eax
  int v101; // r10d
  __int128 v102; // [rsp-30h] [rbp-190h]
  __int128 v103; // [rsp-30h] [rbp-190h]
  __int128 v104; // [rsp-20h] [rbp-180h]
  __int16 v105; // [rsp+Ah] [rbp-156h]
  __int16 v106; // [rsp+12h] [rbp-14Eh]
  __int16 v107; // [rsp+1Ah] [rbp-146h]
  __int64 v108; // [rsp+20h] [rbp-140h]
  unsigned __int64 v109; // [rsp+28h] [rbp-138h]
  __int64 v110; // [rsp+30h] [rbp-130h]
  __int64 v111; // [rsp+30h] [rbp-130h]
  __int64 v112; // [rsp+30h] [rbp-130h]
  __int128 v113; // [rsp+30h] [rbp-130h]
  unsigned int v114; // [rsp+30h] [rbp-130h]
  unsigned int v115; // [rsp+40h] [rbp-120h]
  __int64 v116; // [rsp+40h] [rbp-120h]
  unsigned int v117; // [rsp+40h] [rbp-120h]
  __int64 *v118; // [rsp+40h] [rbp-120h]
  __int128 v119; // [rsp+40h] [rbp-120h]
  __int64 *v120; // [rsp+50h] [rbp-110h]
  __int64 *v121; // [rsp+50h] [rbp-110h]
  __int128 v122; // [rsp+50h] [rbp-110h]
  unsigned __int64 v123; // [rsp+B8h] [rbp-A8h]
  unsigned int v124; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v125; // [rsp+C8h] [rbp-98h]
  __int64 v126; // [rsp+D0h] [rbp-90h] BYREF
  int v127; // [rsp+D8h] [rbp-88h]
  __int16 v128; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v129; // [rsp+E8h] [rbp-78h]
  __int128 v130; // [rsp+F0h] [rbp-70h] BYREF
  __int128 v131; // [rsp+100h] [rbp-60h] BYREF
  __m128i v132; // [rsp+110h] [rbp-50h] BYREF
  __int64 v133; // [rsp+120h] [rbp-40h]
  unsigned int v134; // [rsp+128h] [rbp-38h]

  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v132, *a1, *(_QWORD *)(v9 + 64), v7, v8);
    LOWORD(v124) = v132.m128i_i16[4];
    v125 = v133;
  }
  else
  {
    v124 = v5(*a1, *(_QWORD *)(v9 + 64), v7, v8);
    v125 = v81;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *(_QWORD *)(a2 + 40);
  v12 = *(_DWORD *)(v11 + 8);
  v109 = *(_QWORD *)v11;
  v108 = *(_QWORD *)(v11 + 8);
  v13 = *(_QWORD *)v11;
  v126 = v10;
  if ( v10 )
  {
    v115 = v12;
    sub_B96E90((__int64)&v126, v10, 1);
    v12 = v115;
  }
  v14 = *a1;
  v127 = *(_DWORD *)(a2 + 72);
  v110 = v12;
  v15 = (unsigned __int16 *)(*(_QWORD *)(v13 + 48) + 16LL * v12);
  v116 = 16LL * v12;
  sub_2FE6CC0((__int64)&v132, v14, *(_QWORD *)(a1[1] + 64), *v15, *((_QWORD *)v15 + 1));
  v16 = v116;
  switch ( v132.m128i_i8[0] )
  {
    case 0:
    case 2:
      v17 = (_QWORD *)a1[1];
      v18 = v13;
      v19 = v110;
      if ( *(_DWORD *)(a2 + 24) != 458 )
        goto LABEL_7;
      goto LABEL_41;
    case 1:
      v18 = sub_37AE0F0((__int64)a1, v109, v108);
      v17 = (_QWORD *)a1[1];
      v19 = v80;
      if ( *(_DWORD *)(a2 + 24) == 458 )
      {
LABEL_41:
        *((_QWORD *)&v103 + 1) = v19;
        *(_QWORD *)&v103 = v18;
        v20 = sub_340F900(
                v17,
                0x1CAu,
                (__int64)&v126,
                v124,
                v125,
                v16,
                v103,
                *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
      }
      else
      {
LABEL_7:
        v20 = (__int64)sub_33FAF80((__int64)v17, 216, (__int64)&v126, v124, v125, v16, a3);
      }
      goto LABEL_8;
    case 6:
      v58 = (unsigned __int16 *)(*(_QWORD *)(v13 + 48) + v116);
      v59 = *v58;
      v60 = *((_QWORD *)v58 + 1);
      v128 = v59;
      v129 = v60;
      if ( (_WORD)v59 )
      {
        v63 = (unsigned __int16)(v59 - 176) <= 0x34u;
        LOBYTE(v62) = v63;
        v61 = word_4456340[v59 - 1];
      }
      else
      {
        v123 = sub_3007240((__int64)&v128);
        v61 = v123;
        v62 = HIDWORD(v123);
        v63 = BYTE4(v123);
      }
      v64 = v61 >> 1;
      *(_QWORD *)&v130 = 0;
      DWORD2(v130) = 0;
      *(_QWORD *)&v131 = 0;
      DWORD2(v131) = 0;
      sub_375E8D0((__int64)a1, v109, v108, (__int64)&v130, (__int64)&v131);
      v65 = v124;
      v66 = v64;
      if ( (_WORD)v124 )
      {
        if ( (unsigned __int16)(v124 - 17) > 0xD3u )
        {
LABEL_32:
          v112 = v125;
          goto LABEL_33;
        }
        v112 = 0;
        v65 = word_4456580[(unsigned __int16)v124 - 1];
      }
      else
      {
        v82 = sub_30070B0((__int64)&v124);
        v65 = 0;
        v66 = v64;
        if ( !v82 )
          goto LABEL_32;
        v85 = sub_3009970((__int64)&v124, v64, v83, 0, v84);
        v66 = v64;
        v112 = v86;
        v65 = v85;
      }
LABEL_33:
      v67 = v65;
      v68 = *(__int64 **)(a1[1] + 64);
      v132.m128i_i32[0] = v66;
      v132.m128i_i8[4] = v62;
      v118 = v68;
      if ( v63 )
        LOWORD(v69) = sub_2D43AD0(v65, v66);
      else
        LOWORD(v69) = sub_2D43050(v65, v66);
      v72 = 0;
      if ( !(_WORD)v69 )
      {
        v69 = sub_3009450(v118, v67, v112, v132.m128i_i64[0], v70, v71);
        v107 = HIWORD(v69);
        v72 = v95;
      }
      HIWORD(v73) = v107;
      LOWORD(v73) = v69;
      if ( *(_DWORD *)(a2 + 24) == 216 )
      {
        v91 = sub_33FAF80(a1[1], 216, (__int64)&v126, v73, v72, v71, a3);
        v92 = a1[1];
        *(_QWORD *)&v130 = v91;
        DWORD2(v130) = v93;
        *(_QWORD *)&v131 = sub_33FAF80(v92, 216, (__int64)&v126, v73, v72, v94, a3);
      }
      else
      {
        sub_3777990(&v132, a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a3);
        *(_QWORD *)&v119 = v132.m128i_i64[0];
        *(_QWORD *)&v113 = v133;
        *((_QWORD *)&v119 + 1) = v132.m128i_u32[2];
        *((_QWORD *)&v113 + 1) = v134;
        sub_3408380(
          &v132,
          (_QWORD *)a1[1],
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          a3,
          (__int64)&v126);
        *(_QWORD *)&v122 = v133;
        *((_QWORD *)&v104 + 1) = v132.m128i_u32[2];
        *(_QWORD *)&v104 = v132.m128i_i64[0];
        *((_QWORD *)&v122 + 1) = v134;
        *(_QWORD *)&v130 = sub_340F900((_QWORD *)a1[1], 0x1CAu, (__int64)&v126, v73, v72, v74, v130, v119, v104);
        v75 = (_QWORD *)a1[1];
        DWORD2(v130) = v76;
        *(_QWORD *)&v131 = sub_340F900(v75, 0x1CAu, (__int64)&v126, v73, v72, v77, v131, v113, v122);
      }
      DWORD2(v131) = v78;
      v20 = (__int64)sub_3406EB0((_QWORD *)a1[1], 0x9Fu, (__int64)&v126, v124, v125, v79, v130, v131);
      goto LABEL_8;
    case 7:
      v132.m128i_i32[0] = sub_375D5B0((__int64)a1, v109, v108);
      v22 = sub_3805BC0((__int64)(a1 + 181), v132.m128i_i32);
      sub_37593F0((__int64)a1, v22);
      v23 = a1[64] & 1;
      if ( v23 )
      {
        v25 = a1 + 65;
        v26 = 7;
      }
      else
      {
        v24 = *((unsigned int *)a1 + 132);
        v25 = (_QWORD *)a1[65];
        if ( !(_DWORD)v24 )
          goto LABEL_59;
        v26 = v24 - 1;
      }
      v27 = v26 & (37 * *v22);
      v28 = &v25[3 * v27];
      v29 = *(_DWORD *)v28;
      if ( *v22 == *(_DWORD *)v28 )
        goto LABEL_15;
      for ( i = 1; ; i = v101 )
      {
        if ( v29 == -1 )
        {
          if ( v23 )
          {
            v99 = 24;
            goto LABEL_60;
          }
          v24 = *((unsigned int *)a1 + 132);
LABEL_59:
          v99 = 3 * v24;
LABEL_60:
          v28 = &v25[v99];
          break;
        }
        v101 = i + 1;
        v27 = v26 & (i + v27);
        v28 = &v25[3 * v27];
        v29 = *(_DWORD *)v28;
        if ( *v22 == *(_DWORD *)v28 )
          break;
      }
LABEL_15:
      v30 = v28[1];
      v31 = *((unsigned int *)v28 + 4);
      v32 = *(_QWORD *)(v30 + 48) + 16 * v31;
      v33 = v31;
      v34 = *(_WORD *)v32;
      v35 = *(_QWORD *)(v32 + 8);
      v132.m128i_i16[0] = v34;
      v132.m128i_i64[1] = v35;
      if ( !v34 )
      {
        if ( sub_3007100((__int64)&v132) )
          goto LABEL_53;
LABEL_51:
        v117 = sub_3007130((__int64)&v132, 0xFFFFFFFF00000000LL);
        goto LABEL_18;
      }
      if ( (unsigned __int16)(v34 - 176) > 0x34u )
        goto LABEL_17;
LABEL_53:
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      if ( !v132.m128i_i16[0] )
        goto LABEL_51;
      if ( (unsigned __int16)(v132.m128i_i16[0] - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
LABEL_17:
      v117 = word_4456340[v132.m128i_u16[0] - 1];
LABEL_18:
      v36 = *(unsigned __int16 **)(a2 + 48);
      v37 = *v36;
      v38 = *((_QWORD *)v36 + 1);
      v132.m128i_i16[0] = v37;
      v132.m128i_i64[1] = v38;
      if ( (_WORD)v37 )
      {
        if ( (unsigned __int16)(v37 - 17) <= 0xD3u )
        {
          v38 = 0;
          LOWORD(v37) = word_4456580[(unsigned __int16)v37 - 1];
        }
      }
      else
      {
        v114 = v37;
        v87 = sub_30070B0((__int64)&v132);
        LOWORD(v37) = v114;
        if ( v87 )
        {
          LOWORD(v37) = sub_3009970((__int64)&v132, 0xFFFFFFFF00000000LL, v88, v114, v89);
          v38 = v90;
        }
      }
      v39 = (unsigned __int16)v37;
      v120 = *(__int64 **)(a1[1] + 64);
      LOWORD(v40) = sub_2D43050(v37, v117);
      v42 = 0;
      if ( !(_WORD)v40 )
      {
        v40 = sub_3009400(v120, v39, v38, v117, 0);
        v106 = HIWORD(v40);
        v42 = v98;
      }
      HIWORD(v43) = v106;
      LOWORD(v43) = v40;
      sub_33FAF80(a1[1], 216, (__int64)&v126, v43, v42, v41, a3);
      if ( (_WORD)v124 )
      {
        v111 = 0;
        v46 = word_4456580[(unsigned __int16)v124 - 1];
      }
      else
      {
        v46 = sub_3009970((__int64)&v124, v33, v44, v30, v45);
        v111 = v97;
      }
      v47 = v46;
      v121 = *(__int64 **)(a1[1] + 64);
      LOWORD(v48) = sub_2D43050(v46, v117);
      v50 = 0;
      if ( !(_WORD)v48 )
      {
        v48 = sub_3009400(v121, v47, v111, v117, 0);
        v105 = HIWORD(v48);
        v50 = v96;
      }
      HIWORD(v51) = v105;
      LOWORD(v51) = v48;
      v52 = sub_33FAF80(a1[1], 214, (__int64)&v126, v51, v50, v49, a3);
      v54 = v53;
      v55 = v52;
      *(_QWORD *)&v56 = sub_3400EE0(a1[1], 0, (__int64)&v126, 0, a3);
      *((_QWORD *)&v102 + 1) = v54;
      *(_QWORD *)&v102 = v55;
      v20 = (__int64)sub_3406EB0((_QWORD *)a1[1], 0xA1u, (__int64)&v126, v124, v125, v57, v102, v56);
LABEL_8:
      if ( v126 )
        sub_B91220((__int64)&v126, v126);
      return v20;
    default:
      BUG();
  }
}
