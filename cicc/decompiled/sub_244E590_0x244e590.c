// Function: sub_244E590
// Address: 0x244e590
//
void __fastcall sub_244E590(__int64 *a1, __int64 *a2, __int64 a3, int a4)
{
  __int64 v5; // r14
  _QWORD *v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  char v18; // al
  char v19; // r13
  unsigned __int8 *v20; // rbx
  unsigned int *v21; // r12
  unsigned int *v22; // r13
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rax
  char v26; // al
  _QWORD *v27; // rax
  __int64 v28; // r9
  __int64 v29; // r13
  __int64 v30; // r12
  unsigned int *v31; // r12
  unsigned int *v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // esi
  unsigned __int8 *v35; // r12
  __int64 (__fastcall *v36)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v37; // r13
  _QWORD *v38; // rax
  __int64 v39; // rbx
  unsigned int *v40; // r13
  unsigned int *v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  unsigned __int8 v44; // bl
  __int64 v45; // r13
  __int64 v46; // r14
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned __int64 v50; // rax
  unsigned __int8 *v51; // rax
  unsigned __int8 *v52; // r12
  __int64 v53; // r13
  __int64 v54; // rbx
  __int64 v55; // rdx
  unsigned int v56; // esi
  unsigned __int8 *v57; // r13
  __int64 (__fastcall *v58)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v59; // rbx
  __int64 v60; // rsi
  __int64 v61; // rdx
  __int64 v62; // r14
  int *v63; // r12
  size_t v64; // rdx
  size_t v65; // rbx
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // r13
  __int64 v69; // rax
  char v70; // bl
  _QWORD *v71; // rax
  __int64 v72; // r9
  __int64 v73; // r12
  __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // rdx
  unsigned int v77; // esi
  _QWORD *v78; // rax
  __int64 v79; // rbx
  __int64 v80; // r13
  __int64 v81; // r12
  __int64 v82; // rdx
  unsigned int v83; // esi
  _QWORD **v84; // rdx
  int v85; // ecx
  int v86; // eax
  __int64 *v87; // rax
  __int64 v88; // rsi
  unsigned int *v89; // r12
  unsigned int *v90; // rbx
  __int64 v91; // rdx
  unsigned int v92; // esi
  __int64 v93; // r13
  __int64 v94; // r12
  __int64 v95; // rdx
  unsigned int v96; // esi
  unsigned int v97; // eax
  __int64 v98; // rcx
  __int64 v99; // r9
  size_t v100; // rdx
  char *v101; // rax
  __int64 v102; // rdx
  __m128i *v103; // rax
  __m128i *v104; // rax
  unsigned __int64 v105; // rax
  unsigned __int64 v106; // rdi
  __m128i *v107; // rax
  unsigned __int64 *v108; // rax
  size_t v109; // rcx
  unsigned __int8 *v110; // rsi
  __m128i *v111; // rax
  __int64 v112; // rcx
  __int64 v113; // r8
  __int64 v114; // r9
  __int64 v115; // [rsp-8h] [rbp-488h]
  char v116; // [rsp+20h] [rbp-460h]
  __int64 v117; // [rsp+28h] [rbp-458h]
  _QWORD *v118; // [rsp+58h] [rbp-428h]
  __int64 v120; // [rsp+68h] [rbp-418h]
  unsigned __int8 *v121; // [rsp+68h] [rbp-418h]
  __int64 v123; // [rsp+80h] [rbp-400h]
  int *v124; // [rsp+90h] [rbp-3F0h]
  __int64 v125; // [rsp+98h] [rbp-3E8h]
  size_t v126; // [rsp+98h] [rbp-3E8h]
  __int64 v127; // [rsp+A0h] [rbp-3E0h]
  unsigned int v129; // [rsp+B0h] [rbp-3D0h] BYREF
  __int64 (__fastcall **v130)(); // [rsp+B8h] [rbp-3C8h]
  __m128i *v131; // [rsp+C0h] [rbp-3C0h]
  size_t v132; // [rsp+C8h] [rbp-3B8h]
  __m128i v133; // [rsp+D0h] [rbp-3B0h] BYREF
  unsigned __int64 v134[2]; // [rsp+E0h] [rbp-3A0h] BYREF
  _BYTE v135[16]; // [rsp+F0h] [rbp-390h] BYREF
  __m128i v136; // [rsp+100h] [rbp-380h] BYREF
  __m128i v137; // [rsp+110h] [rbp-370h] BYREF
  char v138; // [rsp+120h] [rbp-360h]
  char v139; // [rsp+121h] [rbp-35Fh]
  __m128i v140; // [rsp+130h] [rbp-350h] BYREF
  __m128i v141; // [rsp+140h] [rbp-340h] BYREF
  __int16 v142; // [rsp+150h] [rbp-330h]
  __m128i v143; // [rsp+160h] [rbp-320h] BYREF
  _QWORD v144[2]; // [rsp+170h] [rbp-310h] BYREF
  __int16 v145; // [rsp+180h] [rbp-300h]
  unsigned int *v146; // [rsp+190h] [rbp-2F0h] BYREF
  __int64 v147; // [rsp+198h] [rbp-2E8h]
  _BYTE v148[32]; // [rsp+1A0h] [rbp-2E0h] BYREF
  __int64 v149; // [rsp+1C0h] [rbp-2C0h]
  __int64 v150; // [rsp+1C8h] [rbp-2B8h]
  __int64 v151; // [rsp+1D0h] [rbp-2B0h]
  __int64 v152; // [rsp+1D8h] [rbp-2A8h]
  void **v153; // [rsp+1E0h] [rbp-2A0h]
  void **v154; // [rsp+1E8h] [rbp-298h]
  __int64 v155; // [rsp+1F0h] [rbp-290h]
  int v156; // [rsp+1F8h] [rbp-288h]
  __int16 v157; // [rsp+1FCh] [rbp-284h]
  char v158; // [rsp+1FEh] [rbp-282h]
  __int64 v159; // [rsp+200h] [rbp-280h]
  __int64 v160; // [rsp+208h] [rbp-278h]
  void *v161; // [rsp+210h] [rbp-270h] BYREF
  void *v162; // [rsp+218h] [rbp-268h] BYREF
  __m128i v163; // [rsp+220h] [rbp-260h] BYREF
  __m128i v164; // [rsp+230h] [rbp-250h] BYREF
  char v165; // [rsp+240h] [rbp-240h]
  char v166; // [rsp+241h] [rbp-23Fh]
  __int64 v167; // [rsp+250h] [rbp-230h]
  __int64 v168; // [rsp+258h] [rbp-228h]
  __int64 v169; // [rsp+260h] [rbp-220h]
  __int64 v170; // [rsp+268h] [rbp-218h]
  void **v171; // [rsp+270h] [rbp-210h]
  void **v172; // [rsp+278h] [rbp-208h]
  __int64 v173; // [rsp+280h] [rbp-200h]
  int v174; // [rsp+288h] [rbp-1F8h]
  __int16 v175; // [rsp+28Ch] [rbp-1F4h]
  char v176; // [rsp+28Eh] [rbp-1F2h]
  __int64 v177; // [rsp+290h] [rbp-1F0h]
  __int64 v178; // [rsp+298h] [rbp-1E8h]
  void *v179; // [rsp+2A0h] [rbp-1E0h] BYREF
  void *v180; // [rsp+2A8h] [rbp-1D8h] BYREF
  __m128i v181; // [rsp+2C0h] [rbp-1C0h] BYREF
  __int64 (__fastcall **v182)(); // [rsp+2D0h] [rbp-1B0h] BYREF
  __int64 (__fastcall **v183)(); // [rsp+2D8h] [rbp-1A8h] BYREF
  __int64 v184; // [rsp+2E0h] [rbp-1A0h]
  __int64 v185; // [rsp+2E8h] [rbp-198h]
  unsigned __int64 v186; // [rsp+2F0h] [rbp-190h]
  _BYTE *v187; // [rsp+2F8h] [rbp-188h]
  unsigned __int64 v188; // [rsp+300h] [rbp-180h]
  __int64 v189; // [rsp+308h] [rbp-178h]
  volatile signed __int32 *v190; // [rsp+310h] [rbp-170h] BYREF
  int v191; // [rsp+318h] [rbp-168h]
  unsigned __int64 v192[2]; // [rsp+320h] [rbp-160h] BYREF
  _QWORD v193[2]; // [rsp+330h] [rbp-150h] BYREF
  _QWORD v194[28]; // [rsp+340h] [rbp-140h] BYREF
  __int16 v195; // [rsp+420h] [rbp-60h]
  __int64 v196; // [rsp+428h] [rbp-58h]
  __int64 v197; // [rsp+430h] [rbp-50h]
  __int64 v198; // [rsp+438h] [rbp-48h]
  __int64 v199; // [rsp+440h] [rbp-40h]

  v5 = qword_4FE63B0;
  if ( qword_4FE63B0 )
  {
    if ( &_pthread_key_create )
    {
      v97 = pthread_mutex_lock(&stru_4FE62E0);
      if ( v97 )
        sub_4264C5(v97);
      v5 = qword_4FE63B0;
    }
    v129 = 0;
    v130 = sub_2241E40();
    sub_CB7060((__int64)&v146, (_BYTE *)qword_4FE63A8, v5, (__int64)&v129, 4u);
    if ( v129 )
    {
      v166 = 1;
      v163.m128i_i64[0] = (__int64)" to save mapping file for order file instrumentation\n";
      v140.m128i_i64[0] = (__int64)&qword_4FE63A8;
      v136.m128i_i64[0] = (__int64)"Failed to open ";
      v165 = 3;
      v142 = 260;
      v139 = 1;
      v138 = 3;
      sub_9C6370(&v143, &v136, &v140, v98, v129, v99);
      sub_9C6370(&v181, &v143, &v163, v112, v113, v114);
      sub_C64D30((__int64)&v181, 1u);
    }
    sub_222DF20((__int64)v194);
    v195 = 0;
    v194[27] = 0;
    v194[0] = off_4A06798;
    v181.m128i_i64[0] = (__int64)qword_4A072D8;
    v196 = 0;
    v197 = 0;
    v198 = 0;
    v199 = 0;
    *(__int64 *)((char *)v181.m128i_i64 + qword_4A072D8[-3]) = (__int64)&unk_4A07300;
    v181.m128i_i64[1] = 0;
    sub_222DD70((__int64)v181.m128i_i64 + *(_QWORD *)(v181.m128i_i64[0] - 24), 0);
    v182 = (__int64 (__fastcall **)())qword_4A07288;
    *(__int64 (__fastcall ***)())((char *)&v182 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
    sub_222DD70((__int64)&v182 + (_QWORD)*(v182 - 3), 0);
    v181.m128i_i64[0] = (__int64)qword_4A07328;
    *(__int64 *)((char *)v181.m128i_i64 + qword_4A07328[-3]) = (__int64)&unk_4A07378;
    v181.m128i_i64[0] = (__int64)off_4A073F0;
    v194[0] = off_4A07440;
    v182 = off_4A07418;
    v183 = off_4A07480;
    v184 = 0;
    v185 = 0;
    v186 = 0;
    v187 = 0;
    v188 = 0;
    v189 = 0;
    sub_220A990(&v190);
    v191 = 24;
    v192[1] = 0;
    v192[0] = (unsigned __int64)v193;
    v183 = off_4A07080;
    LOBYTE(v193[0]) = 0;
    sub_222DD70((__int64)v194, (__int64)&v183);
    *(_DWORD *)((char *)&v185 + (_QWORD)*(v182 - 3)) = *(_DWORD *)((_BYTE *)&v185 + (_QWORD)*(v182 - 3)) & 0xFFFFFFB5
                                                     | 8;
    v124 = (int *)sub_BD5D20(a3);
    v126 = v100;
    sub_C7D030(&v163);
    sub_C7D280(v163.m128i_i32, v124, v126);
    sub_C7D290(&v163, &v143);
    sub_223E760((__int64 *)&v182, v143.m128i_i64[0]);
    v101 = (char *)sub_BD5D20(a3);
    v143.m128i_i64[0] = (__int64)v144;
    sub_244E240(v143.m128i_i64, v101, (__int64)&v101[v102]);
    v134[1] = 0;
    v134[0] = (unsigned __int64)v135;
    v135[0] = 0;
    if ( v188 )
    {
      if ( v188 > v186 )
        sub_2241130(v134, 0, 0, v187, v188 - (_QWORD)v187);
      else
        sub_2241130(v134, 0, 0, v187, v186 - (_QWORD)v187);
    }
    else
    {
      sub_2240AE0(v134, v192);
    }
    v103 = (__m128i *)sub_2241130(v134, 0, 0, "MD5 ", 4u);
    v136.m128i_i64[0] = (__int64)&v137;
    if ( (__m128i *)v103->m128i_i64[0] == &v103[1] )
    {
      v137 = _mm_loadu_si128(v103 + 1);
    }
    else
    {
      v136.m128i_i64[0] = v103->m128i_i64[0];
      v137.m128i_i64[0] = v103[1].m128i_i64[0];
    }
    v136.m128i_i64[1] = v103->m128i_i64[1];
    v103->m128i_i64[0] = (__int64)v103[1].m128i_i64;
    v103->m128i_i64[1] = 0;
    v103[1].m128i_i8[0] = 0;
    if ( v136.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v104 = (__m128i *)sub_2241490((unsigned __int64 *)&v136, " ", 1u);
    v140.m128i_i64[0] = (__int64)&v141;
    if ( (__m128i *)v104->m128i_i64[0] == &v104[1] )
    {
      v141 = _mm_loadu_si128(v104 + 1);
    }
    else
    {
      v140.m128i_i64[0] = v104->m128i_i64[0];
      v141.m128i_i64[0] = v104[1].m128i_i64[0];
    }
    v140.m128i_i64[1] = v104->m128i_i64[1];
    v104->m128i_i64[0] = (__int64)v104[1].m128i_i64;
    v104->m128i_i64[1] = 0;
    v104[1].m128i_i8[0] = 0;
    v105 = 15;
    v106 = 15;
    if ( (__m128i *)v140.m128i_i64[0] != &v141 )
      v106 = v141.m128i_i64[0];
    if ( v140.m128i_i64[1] + v143.m128i_i64[1] <= v106 )
      goto LABEL_80;
    if ( (_QWORD *)v143.m128i_i64[0] != v144 )
      v105 = v144[0];
    if ( v140.m128i_i64[1] + v143.m128i_i64[1] <= v105 )
    {
      v111 = (__m128i *)sub_2241130((unsigned __int64 *)&v143, 0, 0, v140.m128i_i64[0], v140.m128i_u64[1]);
      v163.m128i_i64[0] = (__int64)&v164;
      if ( (__m128i *)v111->m128i_i64[0] == &v111[1] )
      {
        v164 = _mm_loadu_si128(v111 + 1);
      }
      else
      {
        v163.m128i_i64[0] = v111->m128i_i64[0];
        v164.m128i_i64[0] = v111[1].m128i_i64[0];
      }
      v163.m128i_i64[1] = v111->m128i_i64[1];
      v111->m128i_i64[0] = (__int64)v111[1].m128i_i64;
      v111->m128i_i64[1] = 0;
      v111[1].m128i_i8[0] = 0;
    }
    else
    {
LABEL_80:
      v107 = (__m128i *)sub_2241490((unsigned __int64 *)&v140, (char *)v143.m128i_i64[0], v143.m128i_u64[1]);
      v163.m128i_i64[0] = (__int64)&v164;
      if ( (__m128i *)v107->m128i_i64[0] == &v107[1] )
      {
        v164 = _mm_loadu_si128(v107 + 1);
      }
      else
      {
        v163.m128i_i64[0] = v107->m128i_i64[0];
        v164.m128i_i64[0] = v107[1].m128i_i64[0];
      }
      v163.m128i_i64[1] = v107->m128i_i64[1];
      v107->m128i_i64[0] = (__int64)v107[1].m128i_i64;
      v107->m128i_i64[1] = 0;
      v107[1].m128i_i8[0] = 0;
    }
    v108 = sub_2240FD0((unsigned __int64 *)&v163, v163.m128i_u64[1], 0, 1u, 10);
    v131 = &v133;
    if ( (unsigned __int64 *)*v108 == v108 + 2 )
    {
      v133 = _mm_loadu_si128((const __m128i *)v108 + 1);
    }
    else
    {
      v131 = (__m128i *)*v108;
      v133.m128i_i64[0] = v108[2];
    }
    v109 = v108[1];
    *((_BYTE *)v108 + 16) = 0;
    v132 = v109;
    *v108 = (unsigned __int64)(v108 + 2);
    v108[1] = 0;
    if ( (__m128i *)v163.m128i_i64[0] != &v164 )
      j_j___libc_free_0(v163.m128i_u64[0]);
    if ( (__m128i *)v140.m128i_i64[0] != &v141 )
      j_j___libc_free_0(v140.m128i_u64[0]);
    if ( (__m128i *)v136.m128i_i64[0] != &v137 )
      j_j___libc_free_0(v136.m128i_u64[0]);
    if ( (_BYTE *)v134[0] != v135 )
      j_j___libc_free_0(v134[0]);
    if ( (_QWORD *)v143.m128i_i64[0] != v144 )
      j_j___libc_free_0(v143.m128i_u64[0]);
    v110 = (unsigned __int8 *)v131;
    sub_CB6200((__int64)&v146, (unsigned __int8 *)v131, v132);
    if ( v131 != &v133 )
    {
      v110 = (unsigned __int8 *)(v133.m128i_i64[0] + 1);
      j_j___libc_free_0((unsigned __int64)v131);
    }
    v181.m128i_i64[0] = (__int64)off_4A073F0;
    v194[0] = off_4A07440;
    v182 = off_4A07418;
    v183 = off_4A07080;
    if ( (_QWORD *)v192[0] != v193 )
    {
      v110 = (unsigned __int8 *)(v193[0] + 1LL);
      j_j___libc_free_0(v192[0]);
    }
    v183 = off_4A07480;
    sub_2209150(&v190);
    v181.m128i_i64[0] = (__int64)qword_4A07328;
    *(__int64 *)((char *)v181.m128i_i64 + qword_4A07328[-3]) = (__int64)&unk_4A07378;
    v182 = (__int64 (__fastcall **)())qword_4A07288;
    *(__int64 (__fastcall ***)())((char *)&v182 + qword_4A07288[-3]) = (__int64 (__fastcall **)())&unk_4A072B0;
    v181.m128i_i64[0] = (__int64)qword_4A072D8;
    *(__int64 *)((char *)v181.m128i_i64 + qword_4A072D8[-3]) = (__int64)&unk_4A07300;
    v181.m128i_i64[1] = 0;
    v194[0] = off_4A06798;
    sub_222E050((__int64)v194);
    sub_CB5B00((int *)&v146, (__int64)v110);
    if ( &_pthread_key_create )
      pthread_mutex_unlock(&stru_4FE62E0);
  }
  v6 = (_QWORD *)*a2;
  v7 = *(_QWORD *)(a3 + 80);
  v118 = (_QWORD *)*a2;
  v8 = v7 - 24;
  if ( !v7 )
    v8 = 0;
  v125 = v8;
  v127 = sub_BCB2D0((_QWORD *)*a2);
  v9 = sub_BCB2B0(v6);
  v10 = *a2;
  v123 = v9;
  v181.m128i_i64[0] = (__int64)"order_file_entry";
  LOWORD(v184) = 259;
  v11 = sub_22077B0(0x50u);
  v12 = v11;
  if ( v11 )
    sub_AA4D50(v11, v10, (__int64)&v181, a3, v125);
  v152 = sub_AA48A0(v12);
  v13 = *a2;
  v153 = &v161;
  v154 = &v162;
  v146 = (unsigned int *)v148;
  v161 = &unk_49DA100;
  v147 = 0x200000000LL;
  v162 = &unk_49DA0B0;
  v149 = v12;
  v150 = v12 + 48;
  LOWORD(v151) = 0;
  v155 = 0;
  v156 = 0;
  v157 = 512;
  v158 = 7;
  v159 = 0;
  v160 = 0;
  v181.m128i_i64[0] = (__int64)"order_file_set";
  LOWORD(v184) = 259;
  v14 = sub_22077B0(0x50u);
  v15 = v14;
  if ( v14 )
    sub_AA4D50(v14, v13, (__int64)&v181, a3, v125);
  v170 = sub_AA48A0(v15);
  v171 = &v179;
  v172 = &v180;
  v163.m128i_i64[0] = (__int64)&v164;
  v163.m128i_i64[1] = 0x200000000LL;
  v179 = &unk_49DA100;
  v180 = &unk_49DA0B0;
  LOWORD(v169) = 0;
  v175 = 512;
  v173 = 0;
  v174 = 0;
  v176 = 7;
  v177 = 0;
  v178 = 0;
  v167 = v15;
  v168 = v15 + 48;
  v136.m128i_i64[0] = sub_ACD640(v127, 0, 0);
  v136.m128i_i64[1] = sub_ACD640(v127, a4, 0);
  LOWORD(v184) = 257;
  v16 = sub_921130(&v146, a1[4], a1[2], &v136, 2, (__int64)&v181, 0);
  v145 = 257;
  v120 = v16;
  v17 = sub_AA4E30(v149);
  v18 = sub_AE5020(v17, v123);
  LOWORD(v184) = 257;
  v19 = v18;
  v20 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A14);
  if ( v20 )
    sub_B4D190((__int64)v20, v123, v120, (__int64)&v181, 0, v19, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, __m128i *, __int64, __int64))*v154 + 2))(
    v154,
    v20,
    &v143,
    v150,
    v151);
  v21 = v146;
  v22 = &v146[4 * (unsigned int)v147];
  if ( v146 != v22 )
  {
    do
    {
      v23 = *((_QWORD *)v21 + 1);
      v24 = *v21;
      v21 += 4;
      sub_B99FD0((__int64)v20, v24, v23);
    }
    while ( v22 != v21 );
  }
  v117 = sub_ACD640(v123, 1, 0);
  v25 = sub_AA4E30(v149);
  v26 = sub_AE5020(v25, *(_QWORD *)(v117 + 8));
  LOWORD(v184) = 257;
  v116 = v26;
  v27 = sub_BD2C40(80, unk_3F10A10);
  v28 = v117;
  v29 = (__int64)v27;
  if ( v27 )
    sub_B4D3C0((__int64)v27, v117, v120, 0, v116, v117, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64, __int64))*v154 + 2))(
    v154,
    v29,
    &v181,
    v150,
    v151,
    v28);
  v30 = 4LL * (unsigned int)v147;
  if ( v146 != &v146[v30] )
  {
    v121 = v20;
    v31 = &v146[v30];
    v32 = v146;
    do
    {
      v33 = *((_QWORD *)v32 + 1);
      v34 = *v32;
      v32 += 4;
      sub_B99FD0(v29, v34, v33);
    }
    while ( v31 != v32 );
    v20 = v121;
  }
  v145 = 257;
  v35 = (unsigned __int8 *)sub_ACD640(v123, 0, 0);
  v36 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v153 + 7);
  if ( v36 != sub_928890 )
  {
    v37 = v36((__int64)v153, 32u, v20, v35);
LABEL_22:
    if ( v37 )
      goto LABEL_23;
    goto LABEL_52;
  }
  if ( *v20 <= 0x15u && *v35 <= 0x15u )
  {
    v37 = sub_AAB310(0x20u, v20, v35);
    goto LABEL_22;
  }
LABEL_52:
  LOWORD(v184) = 257;
  v37 = (__int64)sub_BD2C40(72, unk_3F10FD0);
  if ( v37 )
  {
    v84 = (_QWORD **)*((_QWORD *)v20 + 1);
    v85 = *((unsigned __int8 *)v84 + 8);
    if ( (unsigned int)(v85 - 17) > 1 )
    {
      v88 = sub_BCB2A0(*v84);
    }
    else
    {
      v86 = *((_DWORD *)v84 + 8);
      BYTE4(v134[0]) = (_BYTE)v85 == 18;
      LODWORD(v134[0]) = v86;
      v87 = (__int64 *)sub_BCB2A0(*v84);
      v88 = sub_BCE1B0(v87, v134[0]);
    }
    sub_B523C0(v37, v88, 53, 32, (__int64)v20, (__int64)v35, (__int64)&v181, 0, 0, 0);
  }
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v154 + 2))(v154, v37, &v143, v150, v151);
  v89 = v146;
  v90 = &v146[4 * (unsigned int)v147];
  if ( v146 != v90 )
  {
    do
    {
      v91 = *((_QWORD *)v89 + 1);
      v92 = *v89;
      v89 += 4;
      sub_B99FD0(v37, v92, v91);
    }
    while ( v90 != v89 );
  }
LABEL_23:
  LOWORD(v184) = 257;
  v38 = sub_BD2C40(72, 3u);
  v39 = (__int64)v38;
  if ( v38 )
    sub_B4C9A0((__int64)v38, v15, v125, v37, 3u, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v154 + 2))(v154, v39, &v181, v150, v151);
  v40 = v146;
  v41 = &v146[4 * (unsigned int)v147];
  if ( v146 != v41 )
  {
    do
    {
      v42 = *((_QWORD *)v40 + 1);
      v43 = *v40;
      v40 += 4;
      sub_B99FD0(v39, v43, v42);
    }
    while ( v41 != v40 );
  }
  v44 = -1;
  v45 = sub_ACD640(v127, 1, 0);
  v46 = a1[1];
  v47 = sub_AA4E30(v167);
  v48 = sub_9208B0(v47, *(_QWORD *)(v45 + 8));
  v181.m128i_i64[1] = v49;
  v181.m128i_i64[0] = (unsigned __int64)(v48 + 7) >> 3;
  v50 = sub_CA1930(&v181);
  if ( v50 )
  {
    _BitScanReverse64(&v50, v50);
    v44 = 63 - (v50 ^ 0x3F);
  }
  LOWORD(v184) = 257;
  v51 = (unsigned __int8 *)sub_BD2C40(80, unk_3F148C0);
  v52 = v51;
  if ( v51 )
    sub_B4D750((__int64)v51, 1, v46, v45, v44, 7, 1, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, __m128i *, __int64, __int64))*v172 + 2))(
    v172,
    v52,
    &v181,
    v168,
    v169);
  v53 = v163.m128i_i64[0];
  v54 = v163.m128i_i64[0] + 16LL * v163.m128i_u32[2];
  if ( v163.m128i_i64[0] != v54 )
  {
    do
    {
      v55 = *(_QWORD *)(v53 + 8);
      v56 = *(_DWORD *)v53;
      v53 += 16;
      sub_B99FD0((__int64)v52, v56, v55);
    }
    while ( v54 != v53 );
  }
  v145 = 257;
  v57 = (unsigned __int8 *)sub_ACD640(v127, 0x1FFFF, 0);
  v58 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v171 + 2);
  if ( v58 != sub_9202E0 )
  {
    v59 = v58((__int64)v171, 28u, v52, v57);
    goto LABEL_38;
  }
  if ( *v52 <= 0x15u && *v57 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v59 = sub_AD5570(28, (__int64)v52, v57, 0, 0);
    else
      v59 = sub_AABE40(0x1Cu, v52, v57);
LABEL_38:
    if ( v59 )
      goto LABEL_39;
  }
  LOWORD(v184) = 257;
  v59 = sub_B504D0(28, (__int64)v52, (__int64)v57, (__int64)&v181, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v172 + 2))(v172, v59, &v143, v168, v169);
  v93 = v163.m128i_i64[0];
  v94 = v163.m128i_i64[0] + 16LL * v163.m128i_u32[2];
  if ( v163.m128i_i64[0] != v94 )
  {
    do
    {
      v95 = *(_QWORD *)(v93 + 8);
      v96 = *(_DWORD *)v93;
      v93 += 16;
      sub_B99FD0(v59, v96, v95);
    }
    while ( v94 != v93 );
  }
LABEL_39:
  v140.m128i_i64[0] = sub_ACD640(v127, 0, 0);
  LOWORD(v184) = 257;
  v60 = a1[3];
  v61 = *a1;
  v140.m128i_i64[1] = v59;
  v62 = sub_921130((unsigned int **)&v163, v60, v61, &v140, 2, (__int64)&v181, 0);
  v63 = (int *)sub_BD5D20(a3);
  v65 = v64;
  sub_C7D030(&v181);
  sub_C7D280(v181.m128i_i32, v63, v65);
  sub_C7D290(&v181, &v143);
  v66 = v143.m128i_i64[0];
  v67 = sub_BCB2E0(v118);
  v68 = sub_ACD640(v67, v66, 0);
  v69 = sub_AA4E30(v167);
  v70 = sub_AE5020(v69, *(_QWORD *)(v68 + 8));
  LOWORD(v184) = 257;
  v71 = sub_BD2C40(80, unk_3F10A10);
  v72 = v115;
  v73 = (__int64)v71;
  if ( v71 )
    sub_B4D3C0((__int64)v71, v68, v62, 0, v70, v115, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64, __int64))*v172 + 2))(
    v172,
    v73,
    &v181,
    v168,
    v169,
    v72);
  v74 = v163.m128i_i64[0];
  v75 = v163.m128i_i64[0] + 16LL * v163.m128i_u32[2];
  if ( v163.m128i_i64[0] != v75 )
  {
    do
    {
      v76 = *(_QWORD *)(v74 + 8);
      v77 = *(_DWORD *)v74;
      v74 += 16;
      sub_B99FD0(v73, v77, v76);
    }
    while ( v75 != v74 );
  }
  LOWORD(v184) = 257;
  v78 = sub_BD2C40(72, 1u);
  v79 = (__int64)v78;
  if ( v78 )
    sub_B4C8F0((__int64)v78, v125, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v172 + 2))(v172, v79, &v181, v168, v169);
  v80 = v163.m128i_i64[0];
  v81 = v163.m128i_i64[0] + 16LL * v163.m128i_u32[2];
  if ( v163.m128i_i64[0] != v81 )
  {
    do
    {
      v82 = *(_QWORD *)(v80 + 8);
      v83 = *(_DWORD *)v80;
      v80 += 16;
      sub_B99FD0(v79, v83, v82);
    }
    while ( v81 != v80 );
  }
  nullsub_61();
  v179 = &unk_49DA100;
  nullsub_63();
  if ( (__m128i *)v163.m128i_i64[0] != &v164 )
    _libc_free(v163.m128i_u64[0]);
  nullsub_61();
  v161 = &unk_49DA100;
  nullsub_63();
  if ( v146 != (unsigned int *)v148 )
    _libc_free((unsigned __int64)v146);
}
