// Function: sub_967C50
// Address: 0x967c50
//
__int64 __fastcall sub_967C50(
        int a1,
        const char **a2,
        int a3,
        int *a4,
        __int64 *a5,
        int *a6,
        __int64 *a7,
        int *a8,
        __int64 *a9,
        int *a10,
        __int64 *a11,
        int *a12,
        __int64 *a13,
        _DWORD *a14)
{
  _DWORD *v14; // rbx
  int v15; // edx
  int v16; // ecx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r13
  int v21; // eax
  int v22; // edx
  __int64 v23; // r13
  __int64 v24; // r12
  _QWORD *v25; // rdi
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // rbx
  size_t v30; // r15
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r12
  __int64 *v34; // r14
  __int64 v35; // [rsp+8h] [rbp-6F0h]
  unsigned int v36; // [rsp+1Ch] [rbp-6DCh]
  __int64 v37; // [rsp+20h] [rbp-6D8h]
  __int64 v38; // [rsp+28h] [rbp-6D0h]
  __int64 v39; // [rsp+30h] [rbp-6C8h]
  __int64 v40; // [rsp+38h] [rbp-6C0h] BYREF
  __int64 v41; // [rsp+40h] [rbp-6B8h]
  __int64 v42; // [rsp+48h] [rbp-6B0h]
  int v43; // [rsp+58h] [rbp-6A0h] BYREF
  int v44; // [rsp+60h] [rbp-698h]
  _QWORD v45[2]; // [rsp+68h] [rbp-690h] BYREF
  char v46; // [rsp+78h] [rbp-680h] BYREF
  _QWORD v47[2]; // [rsp+88h] [rbp-670h] BYREF
  char v48; // [rsp+98h] [rbp-660h] BYREF
  _QWORD v49[2]; // [rsp+A8h] [rbp-650h] BYREF
  char v50; // [rsp+B8h] [rbp-640h] BYREF
  _QWORD v51[2]; // [rsp+C8h] [rbp-630h] BYREF
  char v52; // [rsp+D8h] [rbp-620h] BYREF
  _QWORD v53[2]; // [rsp+E8h] [rbp-610h] BYREF
  char v54; // [rsp+F8h] [rbp-600h] BYREF
  _QWORD v55[2]; // [rsp+108h] [rbp-5F0h] BYREF
  char v56; // [rsp+118h] [rbp-5E0h] BYREF
  __int64 v57; // [rsp+128h] [rbp-5D0h]
  __int64 v58; // [rsp+130h] [rbp-5C8h]
  __int64 v59; // [rsp+138h] [rbp-5C0h]
  char v60; // [rsp+140h] [rbp-5B8h]
  int v61; // [rsp+148h] [rbp-5B0h]
  int v62; // [rsp+158h] [rbp-5A0h] BYREF
  __int64 v63; // [rsp+160h] [rbp-598h]
  int *v64; // [rsp+168h] [rbp-590h]
  int *v65; // [rsp+170h] [rbp-588h]
  __int64 v66; // [rsp+178h] [rbp-580h]
  char *v67; // [rsp+180h] [rbp-578h]
  __int64 v68; // [rsp+188h] [rbp-570h]
  char v69; // [rsp+190h] [rbp-568h] BYREF
  char *v70; // [rsp+1A0h] [rbp-558h]
  __int64 v71; // [rsp+1A8h] [rbp-550h]
  char v72; // [rsp+1B0h] [rbp-548h] BYREF
  char *v73; // [rsp+1C0h] [rbp-538h]
  __int64 v74; // [rsp+1C8h] [rbp-530h]
  char v75; // [rsp+1D0h] [rbp-528h] BYREF
  char *v76; // [rsp+1E0h] [rbp-518h]
  __int64 v77; // [rsp+1E8h] [rbp-510h]
  char v78; // [rsp+1F0h] [rbp-508h] BYREF
  char *v79; // [rsp+200h] [rbp-4F8h]
  __int64 v80; // [rsp+208h] [rbp-4F0h]
  char v81; // [rsp+210h] [rbp-4E8h] BYREF
  char *v82; // [rsp+220h] [rbp-4D8h]
  __int64 v83; // [rsp+228h] [rbp-4D0h]
  char v84; // [rsp+230h] [rbp-4C8h] BYREF
  char *v85; // [rsp+240h] [rbp-4B8h]
  __int64 v86; // [rsp+248h] [rbp-4B0h]
  char v87; // [rsp+250h] [rbp-4A8h] BYREF
  char *v88; // [rsp+260h] [rbp-498h]
  __int64 v89; // [rsp+268h] [rbp-490h]
  char v90; // [rsp+270h] [rbp-488h] BYREF
  char *v91; // [rsp+280h] [rbp-478h]
  __int64 v92; // [rsp+288h] [rbp-470h]
  char v93; // [rsp+290h] [rbp-468h] BYREF
  char *v94; // [rsp+2A0h] [rbp-458h]
  __int64 v95; // [rsp+2A8h] [rbp-450h]
  char v96; // [rsp+2B0h] [rbp-448h] BYREF
  char *v97; // [rsp+2C0h] [rbp-438h]
  __int64 v98; // [rsp+2C8h] [rbp-430h]
  char v99; // [rsp+2D0h] [rbp-428h] BYREF
  char *v100; // [rsp+2E0h] [rbp-418h]
  __int64 v101; // [rsp+2E8h] [rbp-410h]
  char v102; // [rsp+2F0h] [rbp-408h] BYREF
  char *v103; // [rsp+300h] [rbp-3F8h]
  __int64 v104; // [rsp+308h] [rbp-3F0h]
  char v105; // [rsp+310h] [rbp-3E8h] BYREF
  char *v106; // [rsp+320h] [rbp-3D8h]
  __int64 v107; // [rsp+328h] [rbp-3D0h]
  char v108; // [rsp+330h] [rbp-3C8h] BYREF
  char *v109; // [rsp+340h] [rbp-3B8h]
  __int64 v110; // [rsp+348h] [rbp-3B0h]
  char v111; // [rsp+350h] [rbp-3A8h] BYREF
  char *v112; // [rsp+360h] [rbp-398h]
  __int64 v113; // [rsp+368h] [rbp-390h]
  char v114; // [rsp+370h] [rbp-388h] BYREF
  char *v115; // [rsp+380h] [rbp-378h]
  __int64 v116; // [rsp+388h] [rbp-370h]
  char v117; // [rsp+390h] [rbp-368h] BYREF
  char *v118; // [rsp+3A0h] [rbp-358h]
  __int64 v119; // [rsp+3A8h] [rbp-350h]
  char v120; // [rsp+3B0h] [rbp-348h] BYREF
  char *v121; // [rsp+3C0h] [rbp-338h]
  __int64 v122; // [rsp+3C8h] [rbp-330h]
  char v123; // [rsp+3D0h] [rbp-328h] BYREF
  char *v124; // [rsp+3E0h] [rbp-318h]
  __int64 v125; // [rsp+3E8h] [rbp-310h]
  char v126; // [rsp+3F0h] [rbp-308h] BYREF
  char *v127; // [rsp+400h] [rbp-2F8h]
  __int64 v128; // [rsp+408h] [rbp-2F0h]
  char v129; // [rsp+410h] [rbp-2E8h] BYREF
  char *v130; // [rsp+420h] [rbp-2D8h]
  __int64 v131; // [rsp+428h] [rbp-2D0h]
  char v132; // [rsp+430h] [rbp-2C8h] BYREF
  char *v133; // [rsp+440h] [rbp-2B8h]
  __int64 v134; // [rsp+448h] [rbp-2B0h]
  char v135; // [rsp+450h] [rbp-2A8h] BYREF
  char *v136; // [rsp+460h] [rbp-298h]
  __int64 v137; // [rsp+468h] [rbp-290h]
  char v138; // [rsp+470h] [rbp-288h] BYREF
  char *v139; // [rsp+480h] [rbp-278h]
  __int64 v140; // [rsp+488h] [rbp-270h]
  char v141; // [rsp+490h] [rbp-268h] BYREF
  char *v142; // [rsp+4A0h] [rbp-258h]
  __int64 v143; // [rsp+4A8h] [rbp-250h]
  char v144; // [rsp+4B0h] [rbp-248h] BYREF
  char *v145; // [rsp+4C0h] [rbp-238h]
  __int64 v146; // [rsp+4C8h] [rbp-230h]
  char v147; // [rsp+4D0h] [rbp-228h] BYREF
  char *v148; // [rsp+4E0h] [rbp-218h]
  __int64 v149; // [rsp+4E8h] [rbp-210h]
  char v150; // [rsp+4F0h] [rbp-208h] BYREF
  char *v151; // [rsp+500h] [rbp-1F8h]
  __int64 v152; // [rsp+508h] [rbp-1F0h]
  char v153; // [rsp+510h] [rbp-1E8h] BYREF
  char *v154; // [rsp+520h] [rbp-1D8h]
  __int64 v155; // [rsp+528h] [rbp-1D0h]
  char v156; // [rsp+530h] [rbp-1C8h] BYREF
  char *v157; // [rsp+540h] [rbp-1B8h]
  __int64 v158; // [rsp+548h] [rbp-1B0h]
  char v159; // [rsp+550h] [rbp-1A8h] BYREF
  char *v160; // [rsp+560h] [rbp-198h]
  __int64 v161; // [rsp+568h] [rbp-190h]
  char v162; // [rsp+570h] [rbp-188h] BYREF
  char *v163; // [rsp+580h] [rbp-178h]
  __int64 v164; // [rsp+588h] [rbp-170h]
  char v165; // [rsp+590h] [rbp-168h] BYREF
  char *v166; // [rsp+5A0h] [rbp-158h]
  __int64 v167; // [rsp+5A8h] [rbp-150h]
  char v168; // [rsp+5B0h] [rbp-148h] BYREF
  char *v169; // [rsp+5C0h] [rbp-138h]
  __int64 v170; // [rsp+5C8h] [rbp-130h]
  char v171; // [rsp+5D0h] [rbp-128h] BYREF
  char *v172; // [rsp+5E0h] [rbp-118h]
  __int64 v173; // [rsp+5E8h] [rbp-110h]
  char v174; // [rsp+5F0h] [rbp-108h] BYREF
  char *v175; // [rsp+600h] [rbp-F8h]
  __int64 v176; // [rsp+608h] [rbp-F0h]
  char v177; // [rsp+610h] [rbp-E8h] BYREF
  char *v178; // [rsp+620h] [rbp-D8h]
  __int64 v179; // [rsp+628h] [rbp-D0h]
  char v180; // [rsp+630h] [rbp-C8h] BYREF
  char *v181; // [rsp+640h] [rbp-B8h]
  __int64 v182; // [rsp+648h] [rbp-B0h]
  char v183; // [rsp+650h] [rbp-A8h] BYREF
  char *v184; // [rsp+660h] [rbp-98h]
  __int64 v185; // [rsp+668h] [rbp-90h]
  char v186; // [rsp+670h] [rbp-88h] BYREF
  char *v187; // [rsp+680h] [rbp-78h]
  __int64 v188; // [rsp+688h] [rbp-70h]
  char v189; // [rsp+690h] [rbp-68h] BYREF
  char *v190; // [rsp+6A0h] [rbp-58h]
  __int64 v191; // [rsp+6A8h] [rbp-50h]
  char v192; // [rsp+6B0h] [rbp-48h] BYREF

  v45[0] = &v46;
  v14 = a14;
  v47[0] = &v48;
  v49[0] = &v50;
  v51[0] = &v52;
  v53[0] = &v54;
  v55[0] = &v56;
  v44 = 0;
  v45[1] = 0;
  v46 = 0;
  v47[1] = 0;
  v48 = 0;
  v49[1] = 0;
  v50 = 0;
  v51[1] = 0;
  v52 = 0;
  v53[1] = 0;
  v54 = 0;
  v55[1] = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0x1000000000LL;
  v61 = 0;
  v62 = 0;
  v64 = &v62;
  v65 = &v62;
  v67 = &v69;
  v70 = &v72;
  v73 = &v75;
  v76 = &v78;
  v79 = &v81;
  v82 = &v84;
  v85 = &v87;
  v88 = &v90;
  v91 = &v93;
  v94 = &v96;
  v63 = 0;
  v66 = 0;
  v68 = 0;
  v69 = 0;
  v71 = 0;
  v72 = 0;
  v74 = 0;
  v75 = 0;
  v77 = 0;
  v78 = 0;
  v80 = 0;
  v81 = 0;
  v83 = 0;
  v84 = 0;
  v86 = 0;
  v87 = 0;
  v89 = 0;
  v90 = 0;
  v92 = 0;
  v93 = 0;
  v95 = 0;
  v97 = &v99;
  v100 = &v102;
  v103 = &v105;
  v106 = &v108;
  v109 = &v111;
  v112 = &v114;
  v115 = &v117;
  v118 = &v120;
  v121 = &v123;
  v124 = &v126;
  v127 = &v129;
  v96 = 0;
  v98 = 0;
  v99 = 0;
  v101 = 0;
  v102 = 0;
  v104 = 0;
  v105 = 0;
  v107 = 0;
  v108 = 0;
  v110 = 0;
  v111 = 0;
  v113 = 0;
  v114 = 0;
  v116 = 0;
  v117 = 0;
  v119 = 0;
  v120 = 0;
  v122 = 0;
  v123 = 0;
  v125 = 0;
  v126 = 0;
  v128 = 0;
  v130 = &v132;
  v133 = &v135;
  v136 = &v138;
  v139 = &v141;
  v142 = &v144;
  v145 = &v147;
  v148 = &v150;
  v151 = &v153;
  v154 = &v156;
  v157 = &v159;
  v160 = &v162;
  v129 = 0;
  v131 = 0;
  v132 = 0;
  v134 = 0;
  v135 = 0;
  v137 = 0;
  v138 = 0;
  v140 = 0;
  v141 = 0;
  v143 = 0;
  v144 = 0;
  v146 = 0;
  v147 = 0;
  v149 = 0;
  v150 = 0;
  v152 = 0;
  v153 = 0;
  v155 = 0;
  v156 = 0;
  v158 = 0;
  v159 = 0;
  v161 = 0;
  v163 = &v165;
  v166 = &v168;
  v169 = &v171;
  v172 = &v174;
  v175 = &v177;
  v178 = &v180;
  v181 = &v183;
  v184 = &v186;
  v187 = &v189;
  v190 = &v192;
  v162 = 0;
  v164 = 0;
  v165 = 0;
  v167 = 0;
  v168 = 0;
  v170 = 0;
  v171 = 0;
  v173 = 0;
  v174 = 0;
  v176 = 0;
  v177 = 0;
  v179 = 0;
  v180 = 0;
  v182 = 0;
  v183 = 0;
  v185 = 0;
  v186 = 0;
  v188 = 0;
  v189 = 0;
  v191 = 0;
  v192 = 0;
  v36 = sub_967070((__int64)&v43, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, 1);
  *a14 = v43;
  a14[2] = v44;
  sub_2240AE0(a14 + 4, v45);
  sub_2240AE0(a14 + 12, v47);
  sub_2240AE0(a14 + 20, v49);
  sub_2240AE0(a14 + 28, v51);
  sub_2240AE0(a14 + 36, v53);
  sub_2240AE0(a14 + 44, v55);
  v15 = HIDWORD(v58);
  v40 = 0;
  v41 = 0;
  v42 = 0x1000000000LL;
  if ( HIDWORD(v58) )
  {
    sub_C92620(&v40, (unsigned int)v58);
    v17 = HIDWORD(v58);
    v16 = v59;
    v38 = v40;
    v15 = v41;
    v37 = v57;
    HIDWORD(v41) = HIDWORD(v58);
    LODWORD(v42) = v59;
    if ( (_DWORD)v41 )
    {
      v27 = v57;
      v28 = 0;
      v29 = 8LL * (unsigned int)v41 + 8;
      v39 = 8LL * (unsigned int)(v41 - 1);
      v18 = v40;
      while ( 1 )
      {
        v33 = *(_QWORD *)(v27 + v28);
        v34 = (__int64 *)(v18 + v28);
        if ( v33 == -8 || !v33 )
        {
          *v34 = v33;
        }
        else
        {
          v30 = *(_QWORD *)v33;
          v31 = sub_C7D670(*(_QWORD *)v33 + 17LL, 8);
          v32 = v31;
          if ( v30 )
          {
            v35 = v31;
            memcpy((void *)(v31 + 16), (const void *)(v33 + 16), v30);
            v32 = v35;
          }
          *(_BYTE *)(v32 + v30 + 16) = 0;
          *(_QWORD *)v32 = v30;
          *(_DWORD *)(v32 + 8) = *(_DWORD *)(v33 + 8);
          *v34 = v32;
          *(_DWORD *)(v38 + v29) = *(_DWORD *)(v37 + v29);
        }
        v18 = v40;
        v29 += 4;
        if ( v39 == v28 )
          break;
        v27 = v57;
        v28 += 8;
      }
      v14 = a14;
      v15 = v41;
      v17 = HIDWORD(v41);
      v16 = v42;
    }
    else
    {
      v18 = v40;
    }
  }
  else
  {
    v16 = 0;
    v17 = 0;
    v18 = 0;
  }
  v19 = *((_QWORD *)v14 + 26);
  v20 = (unsigned int)v14[54];
  *((_QWORD *)v14 + 26) = v18;
  v14[54] = v15;
  v21 = v14[55];
  v22 = v14[56];
  v40 = v19;
  LODWORD(v41) = v20;
  v14[55] = v17;
  HIDWORD(v41) = v21;
  v14[56] = v16;
  LODWORD(v42) = v22;
  if ( v21 && (_DWORD)v20 )
  {
    v23 = 8 * v20;
    v24 = 0;
    do
    {
      v25 = *(_QWORD **)(v19 + v24);
      if ( v25 && v25 != (_QWORD *)-8LL )
      {
        v17 = *v25 + 17LL;
        sub_C7D6A0(v25, v17, 8);
        v19 = v40;
      }
      v24 += 8;
    }
    while ( v23 != v24 );
  }
  _libc_free(v19, v17);
  *((_BYTE *)v14 + 232) = v60;
  sub_95CDE0((__int64)&v43, v17);
  return v36;
}
