// Function: sub_3833800
// Address: 0x3833800
//
unsigned __int8 *__fastcall sub_3833800(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v3; // r14d
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // rax
  _OWORD **v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // eax
  __int64 v14; // rdx
  unsigned int v15; // ecx
  __int64 v16; // rsi
  const __m128i *v17; // r15
  __m128i v18; // xmm0
  __int32 v19; // eax
  __int64 v20; // r12
  __int64 v21; // rsi
  __int64 v22; // r15
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int16 *v26; // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r11
  __int64 v31; // r13
  unsigned int v32; // r15d
  __int64 v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // r8
  __int128 v39; // rax
  __int64 v40; // r9
  unsigned __int8 *v41; // rax
  unsigned int v42; // edx
  unsigned __int8 *v43; // rax
  __int64 v44; // r8
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r9
  unsigned __int64 v48; // rdx
  __int64 *v49; // rax
  __int64 v50; // rax
  _OWORD *v51; // rdx
  unsigned __int8 *v52; // r14
  __int64 v54; // rsi
  __int64 v55; // rax
  unsigned __int16 v56; // dx
  __int64 v57; // r8
  __int64 v58; // rax
  __int64 v59; // r9
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 *v63; // r13
  unsigned __int64 v64; // rax
  __int64 v65; // r12
  __int64 v66; // rdx
  unsigned __int16 v67; // r15
  char v68; // cl
  unsigned __int64 v69; // rsi
  unsigned int v70; // esi
  unsigned __int16 v71; // ax
  __int64 v72; // r8
  __int64 v73; // r9
  unsigned int v74; // esi
  __int64 v75; // rdx
  unsigned __int64 v76; // rax
  _QWORD *v77; // r14
  __int64 v78; // r13
  __int64 v79; // r8
  __int64 v80; // rcx
  __int128 v81; // rax
  __int64 v82; // r9
  unsigned __int8 *v83; // rax
  __int64 v84; // r10
  __int64 v85; // rdx
  __int64 v86; // r15
  unsigned __int8 *v87; // r14
  __int128 v88; // rax
  __int64 v89; // r9
  int v90; // r9d
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rdx
  unsigned int v94; // edx
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // r9
  _QWORD *v98; // r13
  __m128i si128; // xmm2
  __int64 v100; // rdx
  __int64 v101; // rax
  int v102; // r9d
  __int64 v103; // rax
  __m128i v104; // xmm3
  __int64 v105; // rdx
  unsigned __int16 *v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // rdx
  unsigned int v112; // edi
  int v113; // esi
  int v114; // eax
  __int64 v115; // r9
  __int64 v116; // r14
  _QWORD *v117; // r15
  unsigned int v118; // esi
  __int128 v119; // rax
  __int64 v120; // rdx
  __int128 v121; // [rsp-30h] [rbp-1D0h]
  __int128 v122; // [rsp-20h] [rbp-1C0h]
  __int128 v123; // [rsp-10h] [rbp-1B0h]
  __int128 v124; // [rsp-10h] [rbp-1B0h]
  __int64 v125; // [rsp+10h] [rbp-190h]
  __int64 v126; // [rsp+18h] [rbp-188h]
  __int64 v127; // [rsp+20h] [rbp-180h]
  __int64 v128; // [rsp+28h] [rbp-178h]
  __int64 v129; // [rsp+30h] [rbp-170h]
  unsigned int v130; // [rsp+30h] [rbp-170h]
  __int16 v131; // [rsp+32h] [rbp-16Eh]
  unsigned int v132; // [rsp+38h] [rbp-168h]
  __int16 v133; // [rsp+3Ah] [rbp-166h]
  unsigned __int64 v134; // [rsp+40h] [rbp-160h]
  __int64 v135; // [rsp+40h] [rbp-160h]
  __m128i v136; // [rsp+40h] [rbp-160h]
  __int64 v137; // [rsp+50h] [rbp-150h]
  _QWORD *v138; // [rsp+50h] [rbp-150h]
  __int64 v139; // [rsp+50h] [rbp-150h]
  __int64 v140; // [rsp+58h] [rbp-148h]
  unsigned __int32 v142; // [rsp+68h] [rbp-138h]
  __int64 v143; // [rsp+68h] [rbp-138h]
  __int128 v144; // [rsp+70h] [rbp-130h] BYREF
  __int64 v145; // [rsp+80h] [rbp-120h]
  unsigned __int64 v146; // [rsp+88h] [rbp-118h]
  __int64 v147; // [rsp+90h] [rbp-110h] BYREF
  __int64 v148; // [rsp+98h] [rbp-108h]
  __int64 v149; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v150; // [rsp+A8h] [rbp-F8h]
  __int64 v151; // [rsp+B0h] [rbp-F0h] BYREF
  int v152; // [rsp+B8h] [rbp-E8h]
  unsigned __int16 v153; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v154; // [rsp+C8h] [rbp-D8h]
  __int64 v155; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v156; // [rsp+D8h] [rbp-C8h]
  _OWORD *v157; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v158; // [rsp+E8h] [rbp-B8h]
  _OWORD v159[11]; // [rsp+F0h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a2 + 48);
  v6 = *a1;
  v7 = *(_QWORD *)(v5 + 8);
  LOWORD(v147) = *(_WORD *)v5;
  v8 = a1[1];
  v148 = v7;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v9 == sub_2D56A50 )
  {
    v10 = (_OWORD **)v6;
    sub_2FE6CC0((__int64)&v157, v6, *(_QWORD *)(v8 + 64), v147, v148);
    LOWORD(v13) = v158;
    v14 = *(_QWORD *)&v159[0];
    LOWORD(v149) = v158;
    v150 = *(_QWORD *)&v159[0];
  }
  else
  {
    v13 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v9)(v6, *(_QWORD *)(v8 + 64), (unsigned int)v147);
    v10 = &v157;
    LODWORD(v149) = v13;
    v150 = v14;
  }
  if ( (_WORD)v13 )
  {
    v127 = 0;
    LOWORD(v13) = word_4456580[(unsigned __int16)v13 - 1];
  }
  else
  {
    v13 = sub_3009970((__int64)&v149, (__int64)v10, v14, v11, v12);
    v133 = HIWORD(v13);
    v127 = v92;
  }
  HIWORD(v15) = v133;
  LOWORD(v15) = v13;
  v132 = v15;
  v16 = *(_QWORD *)(a2 + 80);
  v151 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v151, v16, 1);
  v17 = *(const __m128i **)(a2 + 40);
  v152 = *(_DWORD *)(a2 + 72);
  v18 = _mm_loadu_si128((const __m128i *)((char *)v17 + 40));
  v137 = v17[2].m128i_i64[1];
  v19 = v17[3].m128i_i32[0];
  v144 = (__int128)v18;
  v142 = v19;
  if ( (_WORD)v147 )
  {
    if ( (unsigned __int16)(v147 - 176) > 0x34u )
      goto LABEL_9;
  }
  else if ( !sub_3007100((__int64)&v147) )
  {
    goto LABEL_9;
  }
  v54 = *a1;
  v55 = *(_QWORD *)(v17->m128i_i64[0] + 48) + 16LL * v17->m128i_u32[2];
  v56 = *(_WORD *)v55;
  v57 = *(_QWORD *)(v55 + 8);
  v136 = _mm_loadu_si128(v17);
  v58 = a1[1];
  v153 = v56;
  v59 = *(_QWORD *)(v58 + 64);
  v154 = v57;
  sub_2FE6CC0((__int64)&v157, v54, v59, v56, v57);
  if ( (_BYTE)v157 != 6 )
  {
    v54 = *a1;
    sub_2FE6CC0((__int64)&v157, *a1, *(_QWORD *)(a1[1] + 64), v153, v154);
    if ( (_BYTE)v157 )
    {
      sub_2FE6CC0((__int64)&v157, *a1, *(_QWORD *)(a1[1] + 64), v153, v154);
      if ( (_BYTE)v157 == 7 )
      {
        v96 = sub_379AB60((__int64)a1, v136.m128i_u64[0], v136.m128i_i64[1]);
        v98 = (_QWORD *)a1[1];
        si128 = _mm_load_si128((const __m128i *)&v144);
        v157 = (_OWORD *)v96;
        v158 = v100;
        v101 = *(_QWORD *)(a2 + 80);
        v159[0] = si128;
        v155 = v101;
        if ( v101 )
          sub_3813810(&v155);
        *((_QWORD *)&v124 + 1) = 2;
        *(_QWORD *)&v124 = &v157;
        LODWORD(v156) = *(_DWORD *)(a2 + 72);
        sub_33FC220(v98, 161, (__int64)&v155, (unsigned int)v147, v148, v97, v124);
      }
      else
      {
        sub_2FE6CC0((__int64)&v157, *a1, *(_QWORD *)(a1[1] + 64), v153, v154);
        if ( (_BYTE)v157 != 1 )
        {
          if ( (_WORD)v147 )
          {
            if ( (unsigned __int16)(v147 - 176) > 0x34u )
              goto LABEL_55;
          }
          else if ( !sub_3007100((__int64)&v147) )
          {
LABEL_55:
            v17 = *(const __m128i **)(a2 + 40);
LABEL_9:
            v21 = *a1;
            v134 = v17->m128i_i64[0];
            v20 = v134;
            v129 = v17->m128i_i64[1];
            v22 = 16LL * v17->m128i_u32[2];
            sub_2FE6CC0(
              (__int64)&v157,
              *a1,
              *(_QWORD *)(a1[1] + 64),
              *(unsigned __int16 *)(v22 + *(_QWORD *)(v134 + 48)),
              *(_QWORD *)(v22 + *(_QWORD *)(v134 + 48) + 8));
            if ( (_BYTE)v157 == 1 )
            {
              v21 = v134;
              v20 = sub_37AE0F0((__int64)a1, v134, v129);
              v22 = 16LL * v94;
            }
            v26 = (unsigned __int16 *)(*(_QWORD *)(v20 + 48) + v22);
            LODWORD(v27) = *v26;
            v28 = *((_QWORD *)v26 + 1);
            LOWORD(v155) = v27;
            v156 = v28;
            if ( (_WORD)v27 )
            {
              v126 = 0;
              LOWORD(v27) = word_4456580[(int)v27 - 1];
            }
            else
            {
              v27 = sub_3009970((__int64)&v155, v21, v28, v23, v24);
              v128 = v27;
              v126 = v91;
            }
            v29 = v128;
            LOWORD(v29) = v27;
            if ( (_WORD)v147 )
            {
              if ( (unsigned __int16)(v147 - 176) > 0x34u )
              {
LABEL_15:
                v130 = word_4456340[(unsigned __int16)v147 - 1];
                goto LABEL_16;
              }
            }
            else if ( !sub_3007100((__int64)&v147) )
            {
              goto LABEL_44;
            }
            sub_CA17B0(
              "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, u"
              "se EVT::getVectorElementCount() instead");
            if ( (_WORD)v147 )
            {
              if ( (unsigned __int16)(v147 - 176) <= 0x34u )
                sub_CA17B0(
                  "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be droppe"
                  "d, use MVT::getVectorElementCount() instead");
              goto LABEL_15;
            }
LABEL_44:
            v130 = sub_3007130((__int64)&v147, v29);
LABEL_16:
            v157 = v159;
            v158 = 0x800000000LL;
            if ( v130 > 8 )
            {
              sub_C8D5F0((__int64)&v157, v159, v130, 0x10u, v24, v25);
            }
            else if ( !v130 )
            {
              v51 = v159;
              v50 = 0;
LABEL_23:
              *((_QWORD *)&v123 + 1) = v50;
              *(_QWORD *)&v123 = v51;
              v52 = sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v151, v149, v150, a1[1], v123);
              if ( v157 != v159 )
                _libc_free((unsigned __int64)v157);
              goto LABEL_25;
            }
            v30 = v2;
            v31 = v137;
            v32 = 0;
            v125 = v142;
            v143 = 16LL * v142;
            do
            {
              v33 = *(_QWORD *)(v31 + 48) + v143;
              LOWORD(v30) = *(_WORD *)v33;
              v138 = (_QWORD *)a1[1];
              v135 = v30;
              v34 = sub_3400BD0((__int64)v138, v32, (__int64)&v151, (unsigned int)v30, *(_QWORD *)(v33 + 8), 0, v18, 0);
              *(_QWORD *)&v144 = v31;
              v36 = v35;
              v37 = *(_QWORD *)(v31 + 48) + v143;
              LOWORD(v3) = *(_WORD *)v37;
              *((_QWORD *)&v122 + 1) = v36;
              *(_QWORD *)&v122 = v34;
              v38 = *(_QWORD *)(v37 + 8);
              *((_QWORD *)&v144 + 1) = v125 | *((_QWORD *)&v144 + 1) & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v39 = sub_3406EB0(
                                  v138,
                                  0x38u,
                                  (__int64)&v151,
                                  v3,
                                  v38,
                                  v36,
                                  __PAIR128__(*((unsigned __int64 *)&v144 + 1), v31),
                                  v122);
              v41 = sub_3406EB0(
                      (_QWORD *)a1[1],
                      0x9Eu,
                      (__int64)&v151,
                      (unsigned int)v29,
                      v126,
                      v40,
                      *(_OWORD *)*(_QWORD *)(a2 + 40),
                      v39);
              v43 = sub_33FAFB0(a1[1], (__int64)v41, v42, (__int64)&v151, v132, v127, v18);
              v30 = v135;
              v44 = (__int64)v43;
              v45 = (unsigned int)v158;
              v47 = v46;
              v48 = (unsigned int)v158 + 1LL;
              if ( v48 > HIDWORD(v158) )
              {
                v139 = v44;
                v140 = v47;
                sub_C8D5F0((__int64)&v157, v159, v48, 0x10u, v44, v47);
                v45 = (unsigned int)v158;
                v30 = v135;
                v44 = v139;
                v47 = v140;
              }
              v49 = (__int64 *)&v157[v45];
              ++v32;
              *v49 = v44;
              v49[1] = v47;
              v50 = (unsigned int)(v158 + 1);
              LODWORD(v158) = v158 + 1;
            }
            while ( v32 != v130 );
            v51 = v157;
            goto LABEL_23;
          }
          sub_C64ED0("Unable to promote scalable types using BUILD_VECTOR", 1u);
        }
        v103 = sub_37AE0F0((__int64)a1, v136.m128i_u64[0], v136.m128i_i64[1]);
        v104 = _mm_load_si128((const __m128i *)&v144);
        v157 = (_OWORD *)v103;
        v158 = v105;
        v159[0] = v104;
        v106 = (unsigned __int16 *)(*(_QWORD *)(v103 + 48) + 16LL * (unsigned int)v105);
        v107 = *v106;
        v108 = *((_QWORD *)v106 + 1);
        LOWORD(v155) = v107;
        v156 = v108;
        v112 = sub_3281170(&v155, v136.m128i_i64[0], v107, v109, v110);
        if ( (_WORD)v149 )
        {
          v113 = word_4456340[(unsigned __int16)v149 - 1];
          if ( (unsigned __int16)(v149 - 176) > 0x34u )
            LOWORD(v114) = sub_2D43050(v112, v113);
          else
            LOWORD(v114) = sub_2D43AD0(v112, v113);
          v116 = 0;
        }
        else
        {
          v114 = sub_3009490((unsigned __int16 *)&v149, v112, v111);
          v131 = HIWORD(v114);
          v116 = v120;
        }
        HIWORD(v118) = v131;
        v117 = (_QWORD *)a1[1];
        *((_QWORD *)&v119 + 1) = 2;
        LOWORD(v118) = v114;
        *(_QWORD *)&v119 = &v157;
        v155 = *(_QWORD *)(a2 + 80);
        if ( v155 )
        {
          v144 = v119;
          sub_3813810(&v155);
          v119 = v144;
        }
        LODWORD(v156) = *(_DWORD *)(a2 + 72);
        sub_33FC220(v117, 161, (__int64)&v155, v118, v116, v115, v119);
      }
      sub_9C6650(&v155);
      v52 = sub_33FAF80(a1[1], 215, (__int64)&v151, (unsigned int)v149, v150, v102, v18);
      goto LABEL_25;
    }
  }
  v63 = *(__int64 **)(a1[1] + 64);
  LOWORD(v64) = v153;
  if ( v153 )
  {
    v65 = 0;
    v66 = v153 - 1;
    v67 = word_4456580[v66];
LABEL_33:
    v68 = (unsigned __int16)(v64 - 176) <= 0x34u;
    LODWORD(v69) = word_4456340[v66];
    LOBYTE(v64) = v68;
    goto LABEL_34;
  }
  v67 = sub_3009970((__int64)&v153, v54, v60, v61, v62);
  LOWORD(v64) = v153;
  v65 = v93;
  if ( v153 )
  {
    v66 = v153 - 1;
    goto LABEL_33;
  }
  v69 = sub_3007240((__int64)&v153);
  v64 = HIDWORD(v69);
  v146 = v69;
  v68 = BYTE4(v69);
LABEL_34:
  v70 = (unsigned int)v69 >> 1;
  BYTE4(v157) = v64;
  LODWORD(v157) = v70;
  if ( v68 )
    v71 = sub_2D43AD0(v67, v70);
  else
    v71 = sub_2D43050(v67, v70);
  if ( v71 )
  {
    LOWORD(v157) = v71;
    v158 = 0;
  }
  else
  {
    v71 = sub_3009450(v63, v67, v65, (__int64)v157, v72, v73);
    LOWORD(v157) = v71;
    v158 = v95;
    if ( !v71 )
    {
      v145 = sub_3007240((__int64)&v157);
      v74 = v145;
      goto LABEL_39;
    }
  }
  v74 = word_4456340[v71 - 1];
LABEL_39:
  v75 = *(_QWORD *)(v137 + 96);
  v76 = *(_QWORD *)(v75 + 24);
  if ( *(_DWORD *)(v75 + 32) > 0x40u )
    v76 = *(_QWORD *)v76;
  v77 = (_QWORD *)a1[1];
  v78 = 16LL * v142;
  v79 = *(_QWORD *)(*(_QWORD *)(v137 + 48) + v78 + 8);
  v80 = *(unsigned __int16 *)(*(_QWORD *)(v137 + 48) + v78);
  *(_QWORD *)&v144 = v76 % v74;
  *(_QWORD *)&v81 = sub_3400BD0((__int64)v77, v74 * (v76 / v74), (__int64)&v151, v80, v79, 0, v18, 0);
  v83 = sub_3406EB0(v77, 0xA1u, (__int64)&v151, (unsigned int)v157, v158, v82, *(_OWORD *)&v136, v81);
  v84 = v144;
  v86 = v85;
  v87 = v83;
  *(_QWORD *)&v144 = a1[1];
  *(_QWORD *)&v88 = sub_3400BD0(
                      v144,
                      v84,
                      (__int64)&v151,
                      *(unsigned __int16 *)(*(_QWORD *)(v137 + 48) + v78),
                      *(_QWORD *)(*(_QWORD *)(v137 + 48) + v78 + 8),
                      0,
                      v18,
                      0);
  *((_QWORD *)&v121 + 1) = v86;
  *(_QWORD *)&v121 = v87;
  sub_3406EB0((_QWORD *)v144, 0xA1u, (__int64)&v151, (unsigned int)v147, v148, v89, v121, v88);
  v52 = sub_33FAF80(a1[1], 215, (__int64)&v151, (unsigned int)v149, v150, v90, v18);
LABEL_25:
  if ( v151 )
    sub_B91220((__int64)&v151, v151);
  return v52;
}
