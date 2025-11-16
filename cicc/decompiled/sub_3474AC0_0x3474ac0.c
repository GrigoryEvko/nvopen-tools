// Function: sub_3474AC0
// Address: 0x3474ac0
//
__int64 __fastcall sub_3474AC0(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        unsigned __int8 *a8,
        unsigned __int64 a9,
        __int128 a10)
{
  int v10; // ebx
  unsigned __int16 *v11; // rax
  unsigned int v12; // r14d
  unsigned int v13; // r12d
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int v18; // eax
  char *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // r12d
  unsigned int v24; // r12d
  __int64 v26; // rsi
  __int128 v27; // rax
  __int64 v28; // r9
  unsigned int v29; // edx
  __int128 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdx
  __int128 v33; // rax
  __int64 v34; // r9
  __int128 v35; // rax
  __int64 v36; // r9
  unsigned __int8 *v37; // rax
  unsigned int v38; // edx
  __int128 v39; // rax
  __int64 v40; // r9
  unsigned int v41; // edx
  __int64 v42; // r15
  __int64 v43; // rax
  unsigned int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // r15
  unsigned int *v47; // r15
  __int64 v48; // rdx
  __int64 v49; // r9
  unsigned int v50; // edx
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned __int8 *v53; // r15
  unsigned int v54; // edx
  unsigned __int64 v55; // r11
  __int128 v56; // rax
  __int64 v57; // r9
  __int64 v58; // rdx
  __int128 v59; // rax
  __int64 v60; // r8
  __int64 v61; // r9
  __int128 v62; // rax
  __int64 v63; // r9
  __int128 v64; // rax
  __int64 v65; // r9
  unsigned int v66; // edx
  __int128 v67; // rax
  __int64 v68; // r9
  __int128 v69; // rax
  __int64 v70; // r14
  __int64 v71; // r15
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  unsigned int v79; // ecx
  unsigned __int64 v80; // rax
  __int64 v81; // rcx
  unsigned __int8 *v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  unsigned int v87; // edx
  __int128 v88; // rax
  __int64 v89; // rax
  unsigned int v90; // r15d
  __int64 v91; // rdx
  __int128 v92; // rax
  __int128 v93; // rax
  __int64 v94; // rax
  unsigned int v95; // edx
  __int64 v96; // r9
  unsigned int v97; // edx
  __int128 v98; // rax
  __int64 v99; // r9
  unsigned int v100; // edx
  __int64 v101; // r9
  unsigned int v102; // edx
  __int128 v103; // [rsp-30h] [rbp-240h]
  __int128 v104; // [rsp-20h] [rbp-230h]
  __int128 v105; // [rsp-20h] [rbp-230h]
  __int128 v106; // [rsp-20h] [rbp-230h]
  __int128 v107; // [rsp-10h] [rbp-220h]
  __int128 v108; // [rsp-10h] [rbp-220h]
  __int128 v109; // [rsp-10h] [rbp-220h]
  __int128 v110; // [rsp-10h] [rbp-220h]
  __int128 v111; // [rsp-10h] [rbp-220h]
  __int128 v112; // [rsp-10h] [rbp-220h]
  __int128 v113; // [rsp-10h] [rbp-220h]
  __int128 v114; // [rsp+0h] [rbp-210h]
  unsigned int v115; // [rsp+10h] [rbp-200h]
  __int64 v116; // [rsp+10h] [rbp-200h]
  __int128 v117; // [rsp+10h] [rbp-200h]
  __int64 v118; // [rsp+20h] [rbp-1F0h]
  unsigned int v119; // [rsp+20h] [rbp-1F0h]
  __int64 v120; // [rsp+20h] [rbp-1F0h]
  __int64 v121; // [rsp+28h] [rbp-1E8h]
  __int64 (__fastcall *v122)(unsigned int *, __int64, __int64, _QWORD, __int64); // [rsp+30h] [rbp-1E0h]
  unsigned int v123; // [rsp+30h] [rbp-1E0h]
  __int128 v124; // [rsp+30h] [rbp-1E0h]
  __int64 v125; // [rsp+30h] [rbp-1E0h]
  unsigned __int64 v126; // [rsp+38h] [rbp-1D8h]
  __int64 v127; // [rsp+38h] [rbp-1D8h]
  __int128 v128; // [rsp+40h] [rbp-1D0h]
  char *v129; // [rsp+58h] [rbp-1B8h]
  __int128 v130; // [rsp+60h] [rbp-1B0h]
  unsigned int v131; // [rsp+70h] [rbp-1A0h]
  unsigned int v132; // [rsp+74h] [rbp-19Ch]
  __int64 v134; // [rsp+88h] [rbp-188h]
  unsigned __int8 *v135; // [rsp+90h] [rbp-180h]
  __int128 v136; // [rsp+90h] [rbp-180h]
  unsigned __int8 *v138; // [rsp+A0h] [rbp-170h]
  unsigned int v139; // [rsp+A0h] [rbp-170h]
  __int128 v140; // [rsp+A0h] [rbp-170h]
  __int128 v141; // [rsp+A0h] [rbp-170h]
  unsigned __int64 v142; // [rsp+A8h] [rbp-168h]
  unsigned __int8 *v143; // [rsp+C0h] [rbp-150h]
  __int64 v144; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v145; // [rsp+168h] [rbp-A8h]
  unsigned __int64 v146; // [rsp+170h] [rbp-A0h] BYREF
  unsigned int v147; // [rsp+178h] [rbp-98h]
  unsigned __int64 v148; // [rsp+180h] [rbp-90h] BYREF
  unsigned int v149; // [rsp+188h] [rbp-88h]
  __int64 v150; // [rsp+190h] [rbp-80h] BYREF
  int v151; // [rsp+198h] [rbp-78h]
  unsigned __int64 v152; // [rsp+1A0h] [rbp-70h] BYREF
  unsigned int v153; // [rsp+1A8h] [rbp-68h]
  __int128 v154; // [rsp+1B0h] [rbp-60h] BYREF
  unsigned __int8 *v155; // [rsp+1C0h] [rbp-50h] BYREF
  unsigned int v156; // [rsp+1C8h] [rbp-48h]
  __int64 v157; // [rsp+1D0h] [rbp-40h]
  unsigned int v158; // [rsp+1D8h] [rbp-38h]

  v10 = *(_DWORD *)(a2 + 24);
  v144 = a4;
  v145 = a5;
  v135 = a8;
  v11 = *(unsigned __int16 **)(a2 + 48);
  v12 = *v11;
  v134 = *((_QWORD *)v11 + 1);
  v13 = 0;
  if ( v10 == 65 || ((v10 - 59) & 0xFFFFFFFD) == 0 )
    return v13;
  v15 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  LOBYTE(v13) = *(_DWORD *)(v15 + 24) == 11 || *(_DWORD *)(v15 + 24) == 35;
  if ( !(_BYTE)v13 )
    return v13;
  v16 = *(_QWORD *)(v15 + 96);
  v18 = *(_DWORD *)(v16 + 32);
  v147 = v18;
  if ( v18 <= 0x40 )
  {
    v19 = *(char **)(v16 + 24);
    v149 = v18;
    v146 = (unsigned __int64)v19;
    v132 = v18 >> 1;
LABEL_5:
    v148 = 0;
    v20 = 1LL << v132;
    goto LABEL_6;
  }
  sub_C43780((__int64)&v146, (const void **)(v16 + 24));
  v22 = v147;
  v149 = v147;
  v132 = v147 >> 1;
  if ( v147 <= 0x40 )
    goto LABEL_5;
  sub_C43690((__int64)&v148, 0, 0);
  v20 = 1LL << v132;
  if ( v149 > 0x40 )
  {
    *(_QWORD *)(v148 + 8LL * (v22 >> 7)) |= v20;
    if ( (int)sub_C49970((__int64)&v146, &v148) >= 0 )
      goto LABEL_16;
LABEL_7:
    if ( (_WORD)v144 == 1 )
    {
      if ( (*((_BYTE *)a1 + 7086) & 0xFB) != 0 )
      {
        v21 = 1;
LABEL_26:
        if ( (a1[125 * v21 + 1619] & 0xFB0000) != 0 )
          goto LABEL_16;
      }
    }
    else
    {
      if ( !(_WORD)v144 || !*(_QWORD *)&a1[2 * (unsigned __int16)v144 + 28] )
        goto LABEL_16;
      if ( (a1[125 * (unsigned __int16)v144 + 1646] & 0xFB0000) != 0 )
      {
        if ( !*(_QWORD *)&a1[2 * (unsigned __int16)v144 + 28] )
          goto LABEL_16;
        v21 = (unsigned __int16)v144;
        goto LABEL_26;
      }
    }
    if ( !sub_33CC5C0(a6) )
    {
      v24 = v147;
      _RDX = v146;
      if ( v147 <= 0x40 )
      {
        if ( v146 > 1 )
        {
          v131 = 0;
          if ( (v146 & 1) == 0 )
          {
            __asm { tzcnt   rdi, rdx }
            v79 = _RDI;
            if ( v147 <= (unsigned int)_RDI )
              v79 = v147;
            v80 = v146 >> v79;
            v131 = v79;
            if ( v147 <= (unsigned int)_RDI )
              v80 = v146 & 1;
            v146 = v80;
          }
          goto LABEL_32;
        }
      }
      else
      {
        v129 = (char *)v146;
        if ( v24 - (unsigned int)sub_C444A0((__int64)&v146) > 0x40 || *(_QWORD *)v129 > 1u )
        {
          v131 = 0;
          if ( (*v129 & 1) == 0 )
          {
            v131 = sub_C44590((__int64)&v146);
            sub_C482E0((__int64)&v146, v131);
          }
LABEL_32:
          v26 = *(_QWORD *)(a2 + 80);
          v150 = v26;
          if ( v26 )
            sub_B96E90((__int64)&v150, v26, 1);
          v151 = *(_DWORD *)(a2 + 72);
          v128 = 0u;
          sub_C4B490((__int64)&v155, (__int64)&v148, (__int64)&v146);
          v13 = v156;
          if ( v156 <= 0x40 )
          {
            LOBYTE(v13) = v155 == (unsigned __int8 *)1;
          }
          else
          {
            v13 = v156 - 1;
            LOBYTE(v13) = v13 == (unsigned int)sub_C444A0((__int64)&v155);
            if ( v155 )
              j_j___libc_free_0_0((unsigned __int64)v155);
          }
          if ( !(_BYTE)v13 )
            goto LABEL_66;
          if ( !a8 )
          {
            sub_34081D0(
              &v155,
              (_QWORD *)a6,
              *(__int128 **)(a2 + 40),
              (__int64)&v150,
              (unsigned int *)&v144,
              (unsigned int *)&v144,
              a7);
            a9 = v156 | a9 & 0xFFFFFFFF00000000LL;
            v135 = v155;
            *(_QWORD *)&a10 = v157;
            *((_QWORD *)&a10 + 1) = v158 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL;
          }
          if ( v131 )
          {
            if ( v10 != 60 )
            {
              sub_F0A5D0((__int64)&v155, v132, v131);
              *(_QWORD *)&v27 = sub_34007B0(a6, (__int64)&v155, (__int64)&v150, v144, v145, 0, a7, 0);
              *((_QWORD *)&v104 + 1) = a9;
              *(_QWORD *)&v104 = v135;
              *(_QWORD *)&v128 = sub_3406EB0(
                                   (_QWORD *)a6,
                                   0xBAu,
                                   (__int64)&v150,
                                   (unsigned int)v144,
                                   v145,
                                   v28,
                                   v104,
                                   v27);
              *((_QWORD *)&v128 + 1) = v29;
              sub_969240((__int64 *)&v155);
            }
            *(_QWORD *)&v30 = sub_3400E40(a6, v132 - v131, v144, v145, (__int64)&v150, a7);
            *(_QWORD *)&v130 = sub_3406EB0((_QWORD *)a6, 0xBEu, (__int64)&v150, (unsigned int)v144, v145, v31, a10, v30);
            *((_QWORD *)&v130 + 1) = v32;
            *(_QWORD *)&v33 = sub_3400E40(a6, v131, v144, v145, (__int64)&v150, a7);
            *((_QWORD *)&v107 + 1) = a9;
            *(_QWORD *)&v107 = v135;
            *(_QWORD *)&v35 = sub_3406EB0((_QWORD *)a6, 0xC0u, (__int64)&v150, (unsigned int)v144, v145, v34, v107, v33);
            v37 = sub_3406EB0((_QWORD *)a6, 0xBBu, (__int64)&v150, (unsigned int)v144, v145, v36, v35, v130);
            a9 = v38 | a9 & 0xFFFFFFFF00000000LL;
            v135 = v37;
            *(_QWORD *)&v39 = sub_3400E40(a6, v131, v144, v145, (__int64)&v150, a7);
            *(_QWORD *)&a10 = sub_3406EB0((_QWORD *)a6, 0xC0u, (__int64)&v150, (unsigned int)v144, v145, v40, a10, v39);
            *((_QWORD *)&a10 + 1) = v41 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL;
          }
          v42 = *(_QWORD *)(a6 + 64);
          v122 = *(__int64 (__fastcall **)(unsigned int *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
          v43 = sub_2E79000(*(__int64 **)(a6 + 40));
          v44 = v122(a1, v43, v42, (unsigned int)v144, v145);
          v46 = v45;
          v123 = v44;
          v115 = v144;
          v118 = v145;
          if ( (unsigned __int8)sub_328A020((__int64)a1, 0x48u, v144, v145, 0) )
          {
            v47 = (unsigned int *)sub_33E5110((__int64 *)a6, v115, v118, v123, v46);
            *((_QWORD *)&v108 + 1) = a9;
            *(_QWORD *)&v108 = v135;
            v116 = v48;
            v138 = sub_3411F20((_QWORD *)a6, 77, (__int64)&v150, v47, v48, v49, v108, a10);
            v119 = v50;
            *(_QWORD *)&v51 = sub_3400BD0(a6, 0, (__int64)&v150, (unsigned int)v144, v145, 0, a7, 0);
            *((_QWORD *)&v109 + 1) = 1;
            *(_QWORD *)&v109 = v138;
            *((_QWORD *)&v103 + 1) = v119;
            *(_QWORD *)&v103 = v138;
            v53 = sub_3412970((_QWORD *)a6, 72, (__int64)&v150, v47, v116, v52, v103, v51, v109);
            v139 = v54;
            v55 = v54;
          }
          else
          {
            *((_QWORD *)&v111 + 1) = a9;
            *(_QWORD *)&v111 = v135;
            *(_QWORD *)&v117 = sub_3406EB0((_QWORD *)a6, 0x38u, (__int64)&v150, v115, v118, v118, v111, a10);
            *((_QWORD *)&v117 + 1) = v87;
            v121 = v87;
            *(_QWORD *)&v88 = sub_33ED040((_QWORD *)a6, 0xCu);
            *((_QWORD *)&v112 + 1) = a9;
            *(_QWORD *)&v112 = v135;
            *((_QWORD *)&v106 + 1) = v121;
            *(_QWORD *)&v106 = v117;
            v89 = sub_340F900((_QWORD *)a6, 0xD0u, (__int64)&v150, v123, v46, v121, v106, v112, v88);
            v90 = v144;
            v127 = v91;
            v125 = v89;
            v120 = v145;
            if ( (unsigned int)sub_3289F80(a1, (unsigned int)v144, v145) == 1 )
            {
              v94 = (__int64)sub_33FB310(a6, v125, v127, (__int64)&v150, v90, v120, a7);
            }
            else
            {
              *(_QWORD *)&v92 = sub_3400BD0(a6, 0, (__int64)&v150, v90, v120, 0, a7, 0);
              v141 = v92;
              *(_QWORD *)&v93 = sub_3400BD0(a6, 1, (__int64)&v150, (unsigned int)v144, v145, 0, a7, 0);
              v94 = sub_3288B20(a6, (int)&v150, v144, v145, v125, v127, v93, v141, 0);
            }
            *((_QWORD *)&v114 + 1) = v95 | v127 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v114 = v94;
            v53 = sub_3406EB0((_QWORD *)a6, 0x38u, (__int64)&v150, (unsigned int)v144, v145, v96, v117, v114);
            v139 = v97;
            v55 = v97 | *((_QWORD *)&v117 + 1) & 0xFFFFFFFF00000000LL;
          }
          v126 = v55;
          if ( v53 )
          {
            sub_C44740((__int64)&v155, (char **)&v146, v132);
            *(_QWORD *)&v56 = sub_34007B0(a6, (__int64)&v155, (__int64)&v150, v144, v145, 0, a7, 0);
            *((_QWORD *)&v105 + 1) = v139 | v126 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v105 = v53;
            *(_QWORD *)&v140 = sub_3406EB0(
                                 (_QWORD *)a6,
                                 0x3Eu,
                                 (__int64)&v150,
                                 (unsigned int)v144,
                                 v145,
                                 v57,
                                 v105,
                                 v56);
            *((_QWORD *)&v140 + 1) = v58;
            if ( v156 > 0x40 && v155 )
              j_j___libc_free_0_0((unsigned __int64)v155);
            *(_QWORD *)&v59 = sub_3400BD0(a6, 0, (__int64)&v150, (unsigned int)v144, v145, 0, a7, 0);
            if ( v10 == 62 )
              goto LABEL_67;
            v124 = v59;
            *((_QWORD *)&v110 + 1) = a9;
            *(_QWORD *)&v110 = v135;
            *(_QWORD *)&v62 = sub_3406EB0((_QWORD *)a6, 0x36u, (__int64)&v150, v12, v134, v61, v110, a10);
            v136 = v62;
            *(_QWORD *)&v64 = sub_3406EB0((_QWORD *)a6, 0x36u, (__int64)&v150, v12, v134, v63, v140, v124);
            *(_QWORD *)&v136 = sub_3406EB0((_QWORD *)a6, 0x39u, (__int64)&v150, v12, v134, v65, v136, v64);
            *((_QWORD *)&v136 + 1) = v66 | *((_QWORD *)&v136 + 1) & 0xFFFFFFFF00000000LL;
            sub_C473B0((__int64)&v152, (__int64)&v146);
            *(_QWORD *)&v67 = sub_34007B0(a6, (__int64)&v152, (__int64)&v150, v12, v134, 0, a7, 0);
            *(_QWORD *)&v69 = sub_3406EB0((_QWORD *)a6, 0x3Au, (__int64)&v150, v12, v134, v68, v136, v67);
            v154 = v69;
            sub_34081D0(&v155, (_QWORD *)a6, &v154, (__int64)&v150, (unsigned int *)&v144, (unsigned int *)&v144, a7);
            v70 = v157;
            v71 = v158;
            sub_3050D50(a3, (__int64)v155, v156, v72, v73, v74);
            sub_3050D50(a3, v70, v71, v75, v76, v77);
            if ( v153 > 0x40 && v152 )
              j_j___libc_free_0_0(v152);
            if ( v10 != 60 )
            {
LABEL_67:
              v81 = v131;
              if ( v131 )
              {
                sub_F0A5D0((__int64)&v155, v132, v131);
                *(_QWORD *)&v98 = sub_3400E40(a6, v131, v144, v145, (__int64)&v150, a7);
                v143 = sub_3406EB0((_QWORD *)a6, 0xBEu, (__int64)&v150, (unsigned int)v144, v145, v99, v140, v98);
                v142 = v100 | *((_QWORD *)&v140 + 1) & 0xFFFFFFFF00000000LL;
                *((_QWORD *)&v113 + 1) = v142;
                *(_QWORD *)&v113 = v143;
                *(_QWORD *)&v140 = sub_3406EB0(
                                     (_QWORD *)a6,
                                     0x38u,
                                     (__int64)&v150,
                                     (unsigned int)v144,
                                     v145,
                                     v101,
                                     v113,
                                     v128);
                *((_QWORD *)&v140 + 1) = v102 | v142 & 0xFFFFFFFF00000000LL;
                sub_969240((__int64 *)&v155);
              }
              sub_3050D50(a3, v140, *((__int64 *)&v140 + 1), v81, v60, v61);
              v82 = sub_3400BD0(a6, 0, (__int64)&v150, (unsigned int)v144, v145, 0, a7, 0);
              sub_3050D50(a3, (__int64)v82, v83, v84, v85, v86);
            }
          }
          else
          {
LABEL_66:
            v13 = 0;
          }
          if ( v150 )
            sub_B91220((__int64)&v150, v150);
          goto LABEL_17;
        }
      }
    }
LABEL_16:
    v13 = 0;
LABEL_17:
    if ( v149 > 0x40 && v148 )
      j_j___libc_free_0_0(v148);
    goto LABEL_20;
  }
LABEL_6:
  v13 = 0;
  v148 |= v20;
  if ( (int)sub_C49970((__int64)&v146, &v148) < 0 )
    goto LABEL_7;
LABEL_20:
  if ( v147 > 0x40 && v146 )
    j_j___libc_free_0_0(v146);
  return v13;
}
