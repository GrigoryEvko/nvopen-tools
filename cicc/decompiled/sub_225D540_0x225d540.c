// Function: sub_225D540
// Address: 0x225d540
//
__int64 __fastcall sub_225D540(__int64 a1, int a2, unsigned int a3, const char **a4, __m128i a5)
{
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  _QWORD *v11; // r13
  char v12; // al
  char *v13; // r12
  unsigned __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rsi
  char *v17; // rax
  __int64 *v18; // rdx
  __int64 *v19; // rcx
  char *v20; // r14
  unsigned int v21; // r15d
  unsigned int (__fastcall *v22)(unsigned __int64, _QWORD); // rax
  char *v23; // rbx
  unsigned __int64 v24; // r8
  __int64 v25; // r12
  __int64 v26; // rbx
  _QWORD *v27; // rdi
  unsigned int v29; // eax
  size_t v30; // rax
  char *v31; // r13
  __int64 v32; // r14
  size_t v33; // rax
  char *v34; // r8
  _QWORD *v35; // rdx
  char v36; // al
  unsigned int (__fastcall *v37)(unsigned __int64, _QWORD); // rax
  char *v38; // r12
  size_t v39; // rax
  int v40; // ebx
  __int64 v41; // rax
  char *v42; // rsi
  __m128i *v43; // rdx
  __m128i v44; // xmm0
  char *v45; // rdx
  char ***v46; // r15
  size_t v47; // rax
  _BYTE *v48; // rdi
  size_t v49; // rdx
  char **v50; // rax
  char **v51; // rdx
  char v52; // al
  char v53; // bl
  size_t v54; // rax
  char *v55; // r8
  __int64 v56; // r9
  size_t v57; // rax
  char *v58; // r13
  __int64 v59; // rax
  char *v60; // rbx
  unsigned __int64 v61; // rdx
  unsigned int (__fastcall *v62)(unsigned __int64, _QWORD); // rax
  char *v63; // r13
  char v64; // bl
  size_t v65; // rax
  char *v66; // r8
  __int64 v67; // r9
  size_t v68; // rax
  char *v69; // r13
  char v70; // al
  char *v71; // r13
  char v72; // bl
  unsigned __int64 *v73; // r14
  unsigned __int64 v74; // rdx
  char v75; // bl
  size_t v76; // rax
  char *v77; // r8
  __int64 v78; // r9
  size_t v79; // rax
  char *v80; // r13
  __int64 v81; // r14
  char v82; // al
  char v83; // bl
  unsigned __int64 v84; // rdx
  unsigned int (__fastcall *v85)(unsigned __int64, _QWORD); // rax
  __m128i *v86; // rdx
  __m128i si128; // xmm0
  __int64 *v88; // [rsp-8h] [rbp-888h]
  unsigned __int64 *v89; // [rsp+0h] [rbp-880h]
  unsigned __int64 *v90; // [rsp+8h] [rbp-878h]
  char *v91; // [rsp+10h] [rbp-870h]
  char *v92; // [rsp+10h] [rbp-870h]
  __int64 v93; // [rsp+10h] [rbp-870h]
  char *v94; // [rsp+10h] [rbp-870h]
  char *v95; // [rsp+10h] [rbp-870h]
  __int64 v96; // [rsp+18h] [rbp-868h]
  __int64 v97; // [rsp+18h] [rbp-868h]
  char *v98; // [rsp+20h] [rbp-860h]
  unsigned __int64 *v100; // [rsp+30h] [rbp-850h]
  char *v101; // [rsp+30h] [rbp-850h]
  char *v102; // [rsp+30h] [rbp-850h]
  unsigned __int64 *v103; // [rsp+30h] [rbp-850h]
  __int64 v104; // [rsp+38h] [rbp-848h]
  unsigned __int64 *v106; // [rsp+40h] [rbp-840h]
  int v107; // [rsp+48h] [rbp-838h]
  int v108; // [rsp+48h] [rbp-838h]
  int v109; // [rsp+48h] [rbp-838h]
  int v110; // [rsp+48h] [rbp-838h]
  char *v111; // [rsp+48h] [rbp-838h]
  char *srca; // [rsp+50h] [rbp-830h]
  unsigned __int8 *src; // [rsp+50h] [rbp-830h]
  char *srcb; // [rsp+50h] [rbp-830h]
  __int64 v115; // [rsp+60h] [rbp-820h]
  __int64 v116; // [rsp+60h] [rbp-820h]
  __int64 v117; // [rsp+60h] [rbp-820h]
  __int64 v118; // [rsp+60h] [rbp-820h]
  unsigned __int64 *v119; // [rsp+98h] [rbp-7E8h] BYREF
  char v120; // [rsp+A2h] [rbp-7DEh] BYREF
  char v121; // [rsp+A3h] [rbp-7DDh] BYREF
  int v122; // [rsp+A4h] [rbp-7DCh] BYREF
  int v123; // [rsp+A8h] [rbp-7D8h] BYREF
  int v124; // [rsp+ACh] [rbp-7D4h] BYREF
  int v125; // [rsp+B0h] [rbp-7D0h] BYREF
  unsigned int v126; // [rsp+B4h] [rbp-7CCh] BYREF
  __int64 v127; // [rsp+B8h] [rbp-7C8h] BYREF
  __int64 v128; // [rsp+C0h] [rbp-7C0h] BYREF
  __int64 v129; // [rsp+C8h] [rbp-7B8h] BYREF
  __int64 v130; // [rsp+D0h] [rbp-7B0h] BYREF
  char *v131; // [rsp+D8h] [rbp-7A8h] BYREF
  char *s; // [rsp+E0h] [rbp-7A0h] BYREF
  char *v133; // [rsp+E8h] [rbp-798h] BYREF
  char *v134; // [rsp+F0h] [rbp-790h] BYREF
  __int64 v135; // [rsp+F8h] [rbp-788h] BYREF
  __int64 v136; // [rsp+100h] [rbp-780h] BYREF
  char *v137; // [rsp+108h] [rbp-778h] BYREF
  int v138[2]; // [rsp+110h] [rbp-770h] BYREF
  char *v139; // [rsp+118h] [rbp-768h] BYREF
  int v140[2]; // [rsp+120h] [rbp-760h] BYREF
  int v141[2]; // [rsp+128h] [rbp-758h] BYREF
  __int64 v142; // [rsp+130h] [rbp-750h] BYREF
  __int64 v143; // [rsp+138h] [rbp-748h] BYREF
  __int64 v144; // [rsp+140h] [rbp-740h] BYREF
  __int64 v145; // [rsp+148h] [rbp-738h] BYREF
  __int64 v146; // [rsp+150h] [rbp-730h] BYREF
  __int64 v147; // [rsp+158h] [rbp-728h] BYREF
  __int64 v148; // [rsp+160h] [rbp-720h] BYREF
  __int64 v149; // [rsp+168h] [rbp-718h] BYREF
  _BYTE *v150; // [rsp+170h] [rbp-710h] BYREF
  __int64 v151; // [rsp+178h] [rbp-708h]
  _QWORD v152[2]; // [rsp+180h] [rbp-700h] BYREF
  char *v153; // [rsp+190h] [rbp-6F0h] BYREF
  size_t v154; // [rsp+198h] [rbp-6E8h]
  _QWORD v155[2]; // [rsp+1A0h] [rbp-6E0h] BYREF
  char **v156; // [rsp+1B0h] [rbp-6D0h] BYREF
  char **v157; // [rsp+1B8h] [rbp-6C8h]
  char *v158; // [rsp+1C0h] [rbp-6C0h] BYREF
  char *v159; // [rsp+1C8h] [rbp-6B8h]
  __m128i *v160; // [rsp+1D0h] [rbp-6B0h]
  __int64 v161; // [rsp+1D8h] [rbp-6A8h]
  __int64 *v162; // [rsp+1E0h] [rbp-6A0h]
  _DWORD v163[4]; // [rsp+1F0h] [rbp-690h] BYREF
  unsigned __int64 v164[2]; // [rsp+200h] [rbp-680h] BYREF
  _BYTE v165[16]; // [rsp+210h] [rbp-670h] BYREF
  unsigned __int64 v166[2]; // [rsp+220h] [rbp-660h] BYREF
  _BYTE v167[16]; // [rsp+230h] [rbp-650h] BYREF
  unsigned __int64 v168[2]; // [rsp+240h] [rbp-640h] BYREF
  _BYTE v169[16]; // [rsp+250h] [rbp-630h] BYREF
  _QWORD *v170; // [rsp+260h] [rbp-620h]
  __int64 v171; // [rsp+268h] [rbp-618h]
  _BYTE v172[16]; // [rsp+270h] [rbp-610h] BYREF
  _QWORD *v173; // [rsp+280h] [rbp-600h]
  __int64 v174; // [rsp+288h] [rbp-5F8h]
  _BYTE v175[16]; // [rsp+290h] [rbp-5F0h] BYREF
  _QWORD *v176; // [rsp+2A0h] [rbp-5E0h]
  __int64 v177; // [rsp+2A8h] [rbp-5D8h]
  _BYTE v178[16]; // [rsp+2B0h] [rbp-5D0h] BYREF
  unsigned __int64 v179; // [rsp+2C0h] [rbp-5C0h]
  __int64 v180; // [rsp+2C8h] [rbp-5B8h]
  __int64 v181; // [rsp+2D0h] [rbp-5B0h]
  void *v182; // [rsp+2E0h] [rbp-5A0h] BYREF
  char v183[240]; // [rsp+2E8h] [rbp-598h] BYREF
  int *v184; // [rsp+3D8h] [rbp-4A8h]
  __int64 v185; // [rsp+3E0h] [rbp-4A0h] BYREF
  bool v186[8]; // [rsp+3E8h] [rbp-498h] BYREF
  __int64 v187; // [rsp+3F0h] [rbp-490h]
  __int64 v188; // [rsp+3F8h] [rbp-488h]
  void *v189; // [rsp+400h] [rbp-480h] BYREF
  char v190[240]; // [rsp+408h] [rbp-478h] BYREF
  int *v191; // [rsp+4F8h] [rbp-388h]
  __int64 v192; // [rsp+500h] [rbp-380h] BYREF
  bool v193[8]; // [rsp+508h] [rbp-378h] BYREF
  __int64 v194; // [rsp+510h] [rbp-370h]
  __int64 v195; // [rsp+518h] [rbp-368h]
  unsigned __int64 v196; // [rsp+520h] [rbp-360h]
  void *v197; // [rsp+530h] [rbp-350h] BYREF
  char v198[240]; // [rsp+538h] [rbp-348h] BYREF
  int *v199; // [rsp+628h] [rbp-258h]
  __int64 v200; // [rsp+630h] [rbp-250h] BYREF
  bool v201[8]; // [rsp+638h] [rbp-248h] BYREF
  __int64 v202; // [rsp+640h] [rbp-240h]
  _QWORD *v203; // [rsp+648h] [rbp-238h]
  unsigned __int64 v204; // [rsp+650h] [rbp-230h]
  unsigned int *v205; // [rsp+658h] [rbp-228h]
  char *v206; // [rsp+660h] [rbp-220h] BYREF
  __int64 v207; // [rsp+668h] [rbp-218h]
  __int64 v208; // [rsp+670h] [rbp-210h]
  __int64 v209; // [rsp+678h] [rbp-208h]
  __m128i *v210; // [rsp+680h] [rbp-200h]
  __int64 v211; // [rsp+688h] [rbp-1F8h]
  __int64 v212; // [rsp+690h] [rbp-1F0h]
  char v213; // [rsp+698h] [rbp-1E8h] BYREF
  char *v214; // [rsp+6A0h] [rbp-1E0h]
  __int64 v215; // [rsp+6A8h] [rbp-1D8h]
  char v216; // [rsp+6B0h] [rbp-1D0h] BYREF
  char *v217; // [rsp+6E0h] [rbp-1A0h]
  __int64 v218; // [rsp+6E8h] [rbp-198h]
  char v219; // [rsp+6F0h] [rbp-190h] BYREF
  char *v220; // [rsp+710h] [rbp-170h]
  __int64 v221; // [rsp+718h] [rbp-168h]
  char v222; // [rsp+720h] [rbp-160h] BYREF
  char *v223; // [rsp+770h] [rbp-110h]
  __int64 v224; // [rsp+778h] [rbp-108h]
  char v225; // [rsp+780h] [rbp-100h] BYREF
  char *v226; // [rsp+820h] [rbp-60h]
  __int64 v227; // [rsp+828h] [rbp-58h]
  char v228; // [rsp+830h] [rbp-50h] BYREF
  __int16 v229; // [rsp+840h] [rbp-40h]
  __int64 v230; // [rsp+848h] [rbp-38h]

  v168[0] = (unsigned __int64)v169;
  v119 = (unsigned __int64 *)a1;
  v164[0] = (unsigned __int64)v165;
  v166[0] = (unsigned __int64)v167;
  v126 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v131 = 0;
  v163[2] = 0;
  v164[1] = 0;
  v165[0] = 0;
  v166[1] = 0;
  v167[0] = 0;
  v168[1] = 0;
  v169[0] = 0;
  v176 = v178;
  v181 = 0x1000000000LL;
  v177 = 0;
  v178[0] = 0;
  v179 = 0;
  v180 = 0;
  v7 = *(_DWORD *)(a1 + 176);
  v170 = v172;
  v171 = 0;
  v172[0] = 0;
  v173 = v175;
  v174 = 0;
  v175[0] = 0;
  if ( !(unsigned int)sub_967C50(
                        a3,
                        a4,
                        v7,
                        &v122,
                        &v127,
                        &v123,
                        &v128,
                        &v124,
                        &v129,
                        &v125,
                        &v130,
                        (int *)&v126,
                        (__int64 *)&v131,
                        v163) )
  {
    if ( v122 != (_DWORD)v142 || v127 != v143 )
    {
      v107 = v122;
      v115 = v127;
      sub_95D500(&v142, &v143);
      LODWORD(v142) = v107;
      v143 = v115;
    }
    if ( v123 != (_DWORD)v144 || v128 != v145 )
    {
      v108 = v123;
      v116 = v128;
      sub_95D500(&v144, &v145);
      LODWORD(v144) = v108;
      v145 = v116;
    }
    if ( v124 != (_DWORD)v146 || v129 != v147 )
    {
      v109 = v124;
      v117 = v129;
      sub_95D500(&v146, &v147);
      LODWORD(v146) = v109;
      v147 = v117;
    }
    if ( v125 != (_DWORD)v148 || v130 != v149 )
    {
      v110 = v125;
      v118 = v130;
      sub_95D500(&v148, &v149);
      LODWORD(v148) = v110;
      v149 = v118;
    }
    v163[0] = a2;
    v182 = &unk_4A08338;
    sub_2258B00((__int64)v183, (__int64)v163);
    v184 = &v122;
    v185 = v127;
    v186[0] = v163[0] == 0;
    v187 = 0;
    v188 = 0;
    v182 = &unk_4A31DF8;
    v8 = sub_22077B0(0x220u);
    v9 = v8;
    if ( v8 )
      sub_30979D0(v8, "nvllc", 5);
    v189 = &unk_4A08338;
    sub_2258B00((__int64)v190, (__int64)v163);
    v191 = &v125;
    v192 = v130;
    v193[0] = v163[0] == 0;
    v194 = 0;
    v195 = v9;
    v189 = &unk_4A31DE0;
    v196 = 0;
    v10 = sub_22077B0(0x220u);
    v11 = (_QWORD *)v10;
    if ( v10 )
    {
      sub_30979D0(v10, "nvopt", 5);
      *v11 = &unk_4A08350;
    }
    v197 = &unk_4A08338;
    sub_2258B00((__int64)v198, (__int64)v163);
    v199 = &v123;
    v202 = 0;
    v200 = v128;
    v201[0] = v163[0] == 0;
    v203 = v11;
    v197 = &unk_4A08378;
    v204 = 0;
    v205 = &v126;
    if ( v122 > 0 )
    {
      v206 = (char *)&v156;
      *(_QWORD *)(__readfsqword(0) - 24) = &v206;
      *(_QWORD *)(__readfsqword(0) - 32) = sub_2257A90;
      if ( !&_pthread_key_create )
        goto LABEL_216;
      v29 = pthread_once(&dword_4FD6B64, init_routine);
      if ( v29 )
        goto LABEL_217;
      nullsub_1744(&v182);
      if ( v188 )
        (*(void (__fastcall **)(__int64, int *, __int64 *, _QWORD))(*(_QWORD *)v188 + 16LL))(v188, v184, &v185, 0);
      sub_2C83470(v186, (unsigned int)*v184, v185, byte_3F871B3);
      v12 = v126;
      if ( (v126 & 0x82) == 0 )
        goto LABEL_20;
    }
    else
    {
      v12 = v126;
      if ( (v126 & 0x82) == 0 )
      {
LABEL_20:
        if ( (v12 & 4) == 0 )
          goto LABEL_21;
LABEL_73:
        v206 = (char *)&v156;
        *(_QWORD *)(__readfsqword(0) - 24) = &v206;
        *(_QWORD *)(__readfsqword(0) - 32) = sub_2257A90;
        if ( &_pthread_key_create )
        {
          v29 = pthread_once(&dword_4FD6B64, init_routine);
          if ( !v29 )
          {
            sub_3098510(&v189);
            if ( v195 )
              (*(void (__fastcall **)(__int64, int *, __int64 *, _QWORD))(*(_QWORD *)v195 + 16LL))(v195, v191, &v192, 0);
            sub_2C83470(v193, (unsigned int)*v191, v192, byte_3F871B3);
            goto LABEL_21;
          }
LABEL_217:
          sub_4264C5(v29);
        }
LABEL_216:
        v29 = -1;
        goto LABEL_217;
      }
    }
    v206 = (char *)&v156;
    *(_QWORD *)(__readfsqword(0) - 24) = &v206;
    *(_QWORD *)(__readfsqword(0) - 32) = sub_2257A90;
    if ( !&_pthread_key_create )
      goto LABEL_216;
    v29 = pthread_once(&dword_4FD6B64, init_routine);
    if ( v29 )
      goto LABEL_217;
    sub_2260610(&v197);
    if ( v203 )
      (*(void (__fastcall **)(_QWORD *, int *, __int64 *, _QWORD))(*v203 + 16LL))(v203, v199, &v200, 0);
    sub_2C83470(v201, (unsigned int)*v199, v200, byte_3F871B3);
    if ( (v126 & 4) == 0 )
    {
LABEL_21:
      s = 0;
      sub_CEAF80((__int64 *)&s);
      v13 = s;
      if ( s )
      {
        v14 = strlen(s);
        if ( v14 > 0x3FFFFFFFFFFFFFFFLL - v119[11] )
          goto LABEL_223;
        sub_2241490(v119 + 10, v13, v14);
        if ( s )
          j_j___libc_free_0_0((unsigned __int64)s);
      }
      v120 = 0;
      v133 = "nvvmCompileProgram";
      v134 = "LibNVVM program compilation.";
      sub_B6EEA0(&v135);
      sub_B6EEA0(&v136);
      v137 = 0;
      v15 = sub_C996C0("NVVM Module Linker", 18, 0, 0);
      v16 = (__int64)&v206;
      v17 = sub_225A270((__int64 *)v119, &v206, v126, &v135, (__int64)v163, a5);
      v20 = v137;
      v137 = v17;
      if ( v20 )
      {
        sub_BA9C10((_QWORD **)v20, (__int64)&v206, (__int64)v18, (__int64)v19);
        v16 = 880;
        j_j___libc_free_0((unsigned __int64)v20);
      }
      v21 = (unsigned int)v206;
      if ( (_DWORD)v206 )
      {
        if ( v15 )
          sub_C9AF60(v15);
        goto LABEL_33;
      }
      if ( v15 )
        sub_C9AF60(v15);
      v22 = (unsigned int (__fastcall *)(unsigned __int64, _QWORD))v119[26];
      if ( v22 )
      {
        v16 = 0;
        if ( v22(v119[27], 0) )
        {
          v21 = 10;
LABEL_33:
          v23 = v137;
          if ( v137 )
          {
            sub_BA9C10((_QWORD **)v137, v16, (__int64)v18, (__int64)v19);
            j_j___libc_free_0((unsigned __int64)v23);
          }
          sub_B6E710(&v136);
          sub_B6E710(&v135);
          v197 = &unk_4A08378;
          if ( v204 )
            j_j___libc_free_0(v204);
          sub_2258D70((__int64)&v197);
          v189 = &unk_4A31DE0;
          if ( v196 )
            j_j___libc_free_0(v196);
          sub_2258D70((__int64)&v189);
          v182 = &unk_4A31DF8;
          sub_2258D70((__int64)&v182);
          goto LABEL_40;
        }
      }
      if ( (v126 & 1) != 0 )
      {
        v30 = 0;
        v31 = v134;
        if ( v134 )
          v30 = strlen(v134);
        v32 = v30;
        v33 = 0;
        v34 = v133;
        if ( v133 )
        {
          srca = v133;
          v33 = strlen(v133);
          v34 = srca;
        }
        sub_CA08F0((__int64 *)&v153, "LNK", 3u, (__int64)"LibNVVM module linking step.", 28, v120, v34, v33, v31, v32);
        if ( v119[14] )
        {
          v211 = 0x100000000LL;
          v212 = (__int64)&v156;
          v156 = &v158;
          v157 = 0;
          LOBYTE(v158) = 0;
          v207 = 0;
          v208 = 0;
          v209 = 0;
          v210 = 0;
          v206 = (char *)&unk_49DD210;
          sub_CB5980((__int64)&v206, 0, 0, 0);
          sub_A3ACE0((__int64)v137, (__int64)&v206, 1u, 0, 0, 0);
          ((void (__fastcall *)(_QWORD, _QWORD, unsigned __int64))v119[14])(
            *(_QWORD *)v212,
            *(_QWORD *)(v212 + 8),
            v119[15]);
          v206 = (char *)&unk_49DD210;
          sub_CB5840((__int64)&v206);
          if ( v156 != &v158 )
            j_j___libc_free_0((unsigned __int64)v156);
        }
        if ( v153 )
          sub_C9E2A0((__int64)v153);
      }
      v16 = *((_QWORD *)v137 + 95);
      v35 = (_QWORD *)*((_QWORD *)v137 + 96);
      v150 = v152;
      sub_2257AB0((__int64 *)&v150, (_BYTE *)v16, (__int64)v35 + v16);
      v36 = v126;
      if ( (v126 & 8) != 0 )
      {
        if ( !v151 )
        {
LABEL_90:
          v19 = &v146;
          v121 = 0;
          v156 = &v133;
          v157 = &v134;
          v158 = &v120;
          v160 = (__m128i *)&v137;
          v161 = (__int64)&v121;
          v159 = (char *)&v146;
          v162 = (__int64 *)&v119;
          if ( (v36 & 0xA0) == 0x20 && !(unsigned __int8)sub_22584A0((__int64)&v156) )
            goto LABEL_195;
          v37 = (unsigned int (__fastcall *)(unsigned __int64, _QWORD))v119[26];
          if ( v37 )
          {
            v16 = 0;
            if ( v37(v119[27], 0) )
            {
LABEL_93:
              v21 = 10;
              goto LABEL_94;
            }
          }
          v52 = v126;
          if ( (v126 & 0x82) != 0 )
          {
            v53 = v120;
            v54 = 0;
            v55 = v134;
            if ( v134 )
            {
              v91 = v134;
              v54 = strlen(v134);
              v55 = v91;
            }
            v56 = v54;
            v57 = 0;
            v58 = v133;
            if ( v133 )
            {
              v92 = v55;
              v96 = v56;
              v57 = strlen(v133);
              v55 = v92;
              v56 = v96;
            }
            sub_CA08F0((__int64 *)v138, "OPT", 3u, (__int64)"LibNVVM optimization step.", 26, v53, v58, v57, v55, v56);
            if ( v119[16] )
            {
              v211 = 0x100000000LL;
              v212 = (__int64)&v153;
              v153 = (char *)v155;
              v154 = 0;
              LOBYTE(v155[0]) = 0;
              v207 = 0;
              v208 = 0;
              v209 = 0;
              v210 = 0;
              v206 = (char *)&unk_49DD210;
              sub_CB5980((__int64)&v206, 0, 0, 0);
              sub_A3ACE0((__int64)v137, (__int64)&v206, 1u, 0, 0, 0);
              ((void (__fastcall *)(_QWORD, _QWORD, unsigned __int64))v119[16])(
                *(_QWORD *)v212,
                *(_QWORD *)(v212 + 8),
                v119[17]);
              v206 = (char *)&unk_49DD210;
              sub_CB5840((__int64)&v206);
              if ( v153 != (char *)v155 )
                j_j___libc_free_0((unsigned __int64)v153);
            }
            v93 = sub_C996C0("NVVM Optimizer", 14, 0, 0);
            v139 = 0;
            v59 = sub_226A3D0(
                    (unsigned int)&v197,
                    (_DWORD)v137,
                    (unsigned int)&v139,
                    (unsigned int)&v136,
                    (unsigned int)&v189,
                    (int)v119 + 48,
                    (__int64)(v119 + 23),
                    (__int64)(v119 + 26));
            v16 = (__int64)v139;
            v60 = (char *)v59;
            v19 = v88;
            if ( v139 )
            {
              v98 = v139;
              v89 = v119;
              v90 = v119 + 10;
              v61 = strlen(v139);
              if ( v61 > 0x3FFFFFFFFFFFFFFFLL - v89[11] )
                goto LABEL_223;
              v16 = (__int64)v98;
              sub_2241490(v90, v98, v61);
              if ( v139 )
                j_j___libc_free_0_0((unsigned __int64)v139);
              v139 = 0;
            }
            v18 = (__int64 *)v119;
            v62 = (unsigned int (__fastcall *)(unsigned __int64, _QWORD))v119[26];
            if ( v62 )
            {
              v16 = 0;
              if ( v62(v119[27], 0) )
              {
                v21 = 10;
LABEL_140:
                if ( v93 )
                  sub_C9AF60(v93);
                if ( *(_QWORD *)v138 )
                  sub_C9E2A0(*(__int64 *)v138);
                goto LABEL_94;
              }
            }
            if ( !v60 )
            {
              v21 = 9;
              goto LABEL_140;
            }
            v63 = v137;
            if ( v60 != v137 )
            {
              v137 = v60;
              if ( v63 )
              {
                sub_BA9C10((_QWORD **)v63, v16, (__int64)v18, (__int64)v19);
                v16 = 880;
                j_j___libc_free_0((unsigned __int64)v63);
              }
            }
            if ( v119[18] )
            {
              v211 = 0x100000000LL;
              v212 = (__int64)&v153;
              v153 = (char *)v155;
              v154 = 0;
              LOBYTE(v155[0]) = 0;
              v207 = 0;
              v208 = 0;
              v209 = 0;
              v210 = 0;
              v206 = (char *)&unk_49DD210;
              sub_CB5980((__int64)&v206, 0, 0, 0);
              sub_A3ACE0((__int64)v137, (__int64)&v206, 1u, 0, 0, 0);
              v16 = *(_QWORD *)(v212 + 8);
              ((void (__fastcall *)(_QWORD, __int64, unsigned __int64))v119[18])(*(_QWORD *)v212, v16, v119[19]);
              v206 = (char *)&unk_49DD210;
              sub_CB5840((__int64)&v206);
              if ( v153 != (char *)v155 )
              {
                v16 = v155[0] + 1LL;
                j_j___libc_free_0((unsigned __int64)v153);
              }
            }
            if ( v93 )
              sub_C9AF60(v93);
            if ( *(_QWORD *)v138 )
              sub_C9E2A0(*(__int64 *)v138);
            v52 = v126;
          }
          if ( v52 < 0 )
          {
            if ( !(unsigned __int8)sub_22584A0((__int64)&v156) )
            {
LABEL_195:
              v21 = 9;
LABEL_94:
              if ( v150 != (_BYTE *)v152 )
              {
                v16 = v152[0] + 1LL;
                j_j___libc_free_0((unsigned __int64)v150);
              }
              goto LABEL_33;
            }
            v52 = v126;
          }
          if ( (v52 & 0x40) != 0 )
          {
            v64 = v120;
            v65 = 0;
            v66 = v134;
            if ( v134 )
            {
              v94 = v134;
              v65 = strlen(v134);
              v66 = v94;
            }
            v67 = v65;
            v68 = 0;
            v69 = v133;
            if ( v133 )
            {
              v95 = v66;
              v97 = v67;
              v68 = strlen(v133);
              v66 = v95;
              v67 = v97;
            }
            sub_CA08F0((__int64 *)v140, "OPTIXIR", 7u, (__int64)"LibNVVM Optix IR step.", 22, v64, v69, v68, v66, v67);
            v16 = (__int64)a4;
            v153 = 0;
            v70 = sub_309ED50(a3, a4, v137, v119 + 6, &v153);
            v71 = v153;
            v72 = v70;
            if ( v153 )
            {
              v100 = v119;
              v73 = v119 + 10;
              v74 = strlen(v153);
              if ( v74 > 0x3FFFFFFFFFFFFFFFLL - v100[11] )
                goto LABEL_223;
              v16 = (__int64)v71;
              sub_2241490(v73, v71, v74);
              if ( v153 )
                j_j___libc_free_0_0((unsigned __int64)v153);
              v153 = 0;
            }
            if ( !v72 )
            {
              if ( *(_QWORD *)v140 )
                sub_C9E2A0(*(__int64 *)v140);
              goto LABEL_195;
            }
            if ( *(_QWORD *)v140 )
              sub_C9E2A0(*(__int64 *)v140);
            v52 = v126;
          }
          if ( (v52 & 4) == 0 )
            goto LABEL_189;
          v75 = v120;
          v76 = 0;
          v77 = v134;
          if ( v134 )
          {
            v101 = v134;
            v76 = strlen(v134);
            v77 = v101;
          }
          v78 = v76;
          v79 = 0;
          v80 = v133;
          if ( v133 )
          {
            v102 = v77;
            v104 = v78;
            v79 = strlen(v133);
            v77 = v102;
            v78 = v104;
          }
          sub_CA08F0((__int64 *)v141, "LLC", 3u, (__int64)"LibNVVM code-generation step.", 29, v75, v80, v79, v77, v78);
          if ( v119[20] )
          {
            v211 = 0x100000000LL;
            v212 = (__int64)&v153;
            v153 = (char *)v155;
            v154 = 0;
            LOBYTE(v155[0]) = 0;
            v207 = 0;
            v208 = 0;
            v209 = 0;
            v210 = 0;
            v206 = (char *)&unk_49DD210;
            sub_CB5980((__int64)&v206, 0, 0, 0);
            sub_A3ACE0((__int64)v137, (__int64)&v206, 1u, 0, 0, 0);
            ((void (__fastcall *)(_QWORD, _QWORD, unsigned __int64))v119[20])(
              *(_QWORD *)v212,
              *(_QWORD *)(v212 + 8),
              v119[21]);
            v206 = (char *)&unk_49DD210;
            sub_CB5840((__int64)&v206);
            if ( v153 != (char *)v155 )
              j_j___libc_free_0((unsigned __int64)v153);
          }
          v81 = sub_C996C0("NVVM CodeGen", 12, 0, 0);
          v206 = 0;
          v82 = sub_3099970(&v189, v137, v119 + 6, &v206, v119 + 26);
          v16 = (__int64)v206;
          v83 = v82;
          if ( !v206 )
          {
LABEL_184:
            if ( !v83 )
            {
              if ( v81 )
                sub_C9AF60(v81);
              if ( *(_QWORD *)v141 )
                sub_C9E2A0(*(__int64 *)v141);
              v21 = 9;
              goto LABEL_94;
            }
            if ( v81 )
              sub_C9AF60(v81);
            if ( *(_QWORD *)v141 )
              sub_C9E2A0(*(__int64 *)v141);
LABEL_189:
            v18 = (__int64 *)v119;
            v85 = (unsigned int (__fastcall *)(unsigned __int64, _QWORD))v119[26];
            if ( !v85 || (v16 = 0, !v85(v119[27], 0)) )
            {
              if ( v121 )
                v21 = 100;
              goto LABEL_94;
            }
            goto LABEL_93;
          }
          v111 = v206;
          v106 = v119;
          v103 = v119 + 10;
          v84 = strlen(v206);
          if ( v84 <= 0x3FFFFFFFFFFFFFFFLL - v106[11] )
          {
            v16 = (__int64)v111;
            sub_2241490(v103, v111, v84);
            if ( v206 )
              j_j___libc_free_0_0((unsigned __int64)v206);
            v206 = 0;
            goto LABEL_184;
          }
LABEL_223:
          sub_4262D8((__int64)"basic_string::append");
        }
      }
      else if ( !v151 )
      {
        v16 = 0;
        v211 = 0x100000000LL;
        v207 = 0;
        v206 = (char *)&unk_49DD210;
        v212 = (__int64)(v119 + 10);
        v208 = 0;
        v209 = 0;
        v210 = 0;
        sub_CB5980((__int64)&v206, 0, 0, 0);
        v86 = v210;
        if ( (unsigned __int64)(v209 - (_QWORD)v210) <= 0x2B )
        {
          v16 = (__int64)"DataLayoutError: Data Layout string is empty";
          sub_CB6200((__int64)&v206, "DataLayoutError: Data Layout string is empty", 0x2Cu);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4363A50);
          v210[2].m128i_i32[2] = 2037674093;
          v86[2].m128i_i64[0] = 0x6520736920676E69LL;
          *v86 = si128;
          v86[1] = _mm_load_si128((const __m128i *)&xmmword_4363A60);
          v210 = (__m128i *)((char *)v210 + 44);
        }
        v206 = (char *)&unk_49DD210;
        sub_CB5840((__int64)&v206);
LABEL_123:
        v21 = 9;
        goto LABEL_94;
      }
      v206 = 0;
      v229 = 768;
      v228 = 0;
      v210 = (__m128i *)&v213;
      v214 = &v216;
      v215 = 0x600000000LL;
      v217 = &v219;
      v218 = 0x400000000LL;
      v220 = &v222;
      v221 = 0xA00000000LL;
      v223 = &v225;
      v224 = 0x800000000LL;
      v207 = 0;
      v208 = 0;
      v209 = 0;
      v211 = 0;
      v212 = 8;
      v226 = &v228;
      v227 = 0;
      v230 = 0;
      sub_AE1EA0((__int64)&v206, (__int64)(v137 + 312));
      v40 = sub_2C74B60(&v150, 1, 1);
      if ( (unsigned __int8)sub_2C747C0(v164)
        || (unsigned __int8)sub_2C747C0(v168)
        || (unsigned __int8)sub_2C747C0(v166) )
      {
        v40 |= sub_2C74AE0(&v206, &v150);
      }
      if ( (_BYTE)v40 )
        sub_BA9520((__int64)v137, v150, v151);
      v16 = (__int64)(v137 + 312);
      sub_2C74F70(v140, v137 + 312, &v135, 1, 1, 0);
      if ( (*(_QWORD *)v140 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      {
        sub_AE4030(&v206, v16);
        v36 = v126;
        goto LABEL_90;
      }
      *(_QWORD *)v140 = *(_QWORD *)v140 & 0xFFFFFFFFFFFFFFFELL | 1;
      v161 = 0x100000000LL;
      v156 = (char **)&unk_49DD210;
      v162 = (__int64 *)(v119 + 10);
      v157 = 0;
      v158 = 0;
      v159 = 0;
      v160 = 0;
      sub_CB5980((__int64)&v156, 0, 0, 0);
      v41 = *(_QWORD *)v140;
      *(_QWORD *)v140 = 0;
      *(_QWORD *)v141 = v41 | 1;
      sub_C64870((__int64)&v153, (__int64 *)v141);
      v42 = v153;
      sub_CB6200((__int64)&v156, (unsigned __int8 *)v153, v154);
      if ( v153 != (char *)v155 )
      {
        v42 = (char *)(v155[0] + 1LL);
        j_j___libc_free_0((unsigned __int64)v153);
      }
      if ( (v141[0] & 1) != 0 || (*(_QWORD *)v141 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(v141, (__int64)v42);
      v43 = v160;
      if ( (unsigned __int64)(v159 - (char *)v160) <= 0x1B )
      {
        sub_CB6200((__int64)&v156, "\nExample valid data layout:\n", 0x1Cu);
        v45 = (char *)v160;
      }
      else
      {
        v44 = _mm_load_si128((const __m128i *)&xmmword_4281920);
        v160[1].m128i_i32[2] = 171603061;
        v43[1].m128i_i64[0] = 0x6F79616C20617461LL;
        *v43 = v44;
        v45 = &v160[1].m128i_i8[12];
        v160 = (__m128i *)((char *)v160 + 28);
      }
      if ( (unsigned __int64)(v159 - v45) <= 7 )
      {
        v46 = (char ***)sub_CB6200((__int64)&v156, "64-bit: ", 8u);
      }
      else
      {
        v46 = &v156;
        *(_QWORD *)v45 = 0x203A7469622D3436LL;
        v160 = (__m128i *)((char *)v160 + 8);
      }
      v16 = (__int64)off_4C5D0A8[0];
      if ( off_4C5D0A8[0] )
      {
        src = (unsigned __int8 *)off_4C5D0A8[0];
        v47 = strlen(off_4C5D0A8[0]);
        v48 = v46[4];
        v16 = (__int64)src;
        v49 = v47;
        v50 = v46[3];
        if ( v49 <= (char *)v50 - v48 )
        {
          if ( v49 )
          {
            srcb = (char *)v49;
            memcpy(v48, (const void *)v16, v49);
            v51 = (char **)&srcb[(_QWORD)v46[4]];
            v46[4] = v51;
            v50 = v46[3];
            v48 = v51;
          }
          goto LABEL_118;
        }
        v46 = (char ***)sub_CB6200((__int64)v46, src, v49);
      }
      v50 = v46[3];
      v48 = v46[4];
LABEL_118:
      if ( v50 == (char **)v48 )
      {
        v16 = (__int64)"\n";
        sub_CB6200((__int64)v46, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v48 = 10;
        v46[4] = (char **)((char *)v46[4] + 1);
      }
      v156 = (char **)&unk_49DD210;
      sub_CB5840((__int64)&v156);
      if ( (v140[0] & 1) != 0 || (*(_QWORD *)v140 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(v140, v16);
      sub_AE4030(&v206, v16);
      goto LABEL_123;
    }
    goto LABEL_73;
  }
  v38 = v131;
  if ( v131 )
  {
    v39 = strlen(v131);
    sub_2241130(v119 + 10, 0, v119[11], v38, v39);
    if ( v131 )
      j_j___libc_free_0_0((unsigned __int64)v131);
  }
  v21 = 7;
LABEL_40:
  v24 = v179;
  if ( HIDWORD(v180) && (_DWORD)v180 )
  {
    v25 = 8LL * (unsigned int)v180;
    v26 = 0;
    do
    {
      v27 = *(_QWORD **)(v24 + v26);
      if ( v27 && v27 != (_QWORD *)-8LL )
      {
        sub_C7D6A0((__int64)v27, *v27 + 17LL, 8);
        v24 = v179;
      }
      v26 += 8;
    }
    while ( v25 != v26 );
  }
  _libc_free(v24);
  if ( v176 != (_QWORD *)v178 )
    j_j___libc_free_0((unsigned __int64)v176);
  if ( v173 != (_QWORD *)v175 )
    j_j___libc_free_0((unsigned __int64)v173);
  if ( v170 != (_QWORD *)v172 )
    j_j___libc_free_0((unsigned __int64)v170);
  if ( (_BYTE *)v168[0] != v169 )
    j_j___libc_free_0(v168[0]);
  if ( (_BYTE *)v166[0] != v167 )
    j_j___libc_free_0(v166[0]);
  if ( (_BYTE *)v164[0] != v165 )
    j_j___libc_free_0(v164[0]);
  sub_95D500(&v148, &v149);
  sub_95D500(&v146, &v147);
  sub_95D500(&v144, &v145);
  sub_95D500(&v142, &v143);
  return v21;
}
