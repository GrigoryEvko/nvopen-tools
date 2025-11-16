// Function: sub_35933D0
// Address: 0x35933d0
//
void __fastcall sub_35933D0(__int64 a1, __int64 a2)
{
  int v3; // eax
  int v4; // eax
  int v5; // eax
  int v6; // eax
  int v7; // eax
  int v8; // eax
  __m128i *v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // eax
  int v12; // eax
  __m128i *v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // eax
  __m128i *v16; // rax
  unsigned __int64 v17; // rdx
  int v18; // eax
  __m128i *v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // eax
  __m128i *v22; // rax
  unsigned __int64 v23; // rdx
  int v24; // eax
  __m128i *v25; // rax
  unsigned __int64 v26; // rdx
  int v27; // eax
  __m128i *v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // rax
  __m128i si128; // xmm0
  int v33; // eax
  __m128i *v34; // rax
  unsigned __int64 v35; // rdx
  int v36; // eax
  int v37; // eax
  int v38; // eax
  int v39; // eax
  int v40; // eax
  _QWORD *v41; // rax
  int v42; // eax
  __int64 v43; // r13
  _BYTE *v44; // rbx
  __int64 v45; // r13
  __int64 v46; // rax
  _BYTE *v47; // rsi
  char *v48; // rcx
  _BYTE *v49; // rax
  size_t v50; // r12
  __int64 v51; // rax
  _BYTE *v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // rdx
  int v56; // eax
  int v57; // eax
  unsigned __int64 v58; // r15
  __int64 v59; // rbx
  __int64 v60; // rax
  unsigned __int64 *v61; // rbx
  _BYTE *v62; // r15
  int v63; // eax
  char **v64; // rsi
  __int64 v65; // rdi
  unsigned __int64 *v66; // r15
  unsigned __int64 *v67; // r13
  unsigned __int64 *v68; // rbx
  unsigned __int64 v69; // rdi
  unsigned __int64 *i; // rbx
  unsigned __int64 v71; // rdi
  unsigned __int64 *v72; // r13
  unsigned __int64 *v73; // rbx
  unsigned __int64 v74; // rdi
  _BYTE *v75; // r8
  unsigned __int64 v76; // r15
  _BYTE *v77; // rbx
  unsigned __int64 *v78; // r12
  _BYTE *v79; // r13
  int v80; // edx
  char **v81; // rsi
  __int64 v82; // rdi
  _BYTE *v83; // r13
  __int64 v84; // rax
  char *v85; // rcx
  _BYTE *v86; // rax
  size_t v87; // r12
  __int64 v88; // rax
  _BYTE *v89; // rsi
  __int64 v90; // rdx
  int v91; // eax
  int v92; // eax
  unsigned __int64 v93; // r15
  __int64 v94; // [rsp+8h] [rbp-A38h]
  unsigned __int64 v95[4]; // [rsp+C0h] [rbp-980h] BYREF
  unsigned __int64 v96[2]; // [rsp+E0h] [rbp-960h] BYREF
  char v97[16]; // [rsp+F0h] [rbp-950h] BYREF
  unsigned __int64 v98[2]; // [rsp+100h] [rbp-940h] BYREF
  _QWORD v99[2]; // [rsp+110h] [rbp-930h] BYREF
  unsigned __int64 v100[2]; // [rsp+120h] [rbp-920h] BYREF
  char v101[16]; // [rsp+130h] [rbp-910h] BYREF
  unsigned __int64 v102[2]; // [rsp+140h] [rbp-900h] BYREF
  char v103[16]; // [rsp+150h] [rbp-8F0h] BYREF
  unsigned __int64 v104[2]; // [rsp+160h] [rbp-8E0h] BYREF
  _QWORD v105[2]; // [rsp+170h] [rbp-8D0h] BYREF
  unsigned __int64 v106[2]; // [rsp+180h] [rbp-8C0h] BYREF
  char v107[16]; // [rsp+190h] [rbp-8B0h] BYREF
  unsigned __int64 v108[2]; // [rsp+1A0h] [rbp-8A0h] BYREF
  _QWORD v109[2]; // [rsp+1B0h] [rbp-890h] BYREF
  unsigned __int64 v110[2]; // [rsp+1C0h] [rbp-880h] BYREF
  _QWORD v111[2]; // [rsp+1D0h] [rbp-870h] BYREF
  unsigned __int64 v112[2]; // [rsp+1E0h] [rbp-860h] BYREF
  _QWORD v113[2]; // [rsp+1F0h] [rbp-850h] BYREF
  unsigned __int64 v114[2]; // [rsp+200h] [rbp-840h] BYREF
  _QWORD v115[2]; // [rsp+210h] [rbp-830h] BYREF
  unsigned __int64 v116[2]; // [rsp+220h] [rbp-820h] BYREF
  _QWORD v117[2]; // [rsp+230h] [rbp-810h] BYREF
  unsigned __int64 v118[2]; // [rsp+240h] [rbp-800h] BYREF
  _QWORD v119[2]; // [rsp+250h] [rbp-7F0h] BYREF
  unsigned __int64 v120[2]; // [rsp+260h] [rbp-7E0h] BYREF
  _QWORD v121[2]; // [rsp+270h] [rbp-7D0h] BYREF
  unsigned __int64 v122[2]; // [rsp+280h] [rbp-7C0h] BYREF
  _QWORD v123[2]; // [rsp+290h] [rbp-7B0h] BYREF
  unsigned __int64 v124[2]; // [rsp+2A0h] [rbp-7A0h] BYREF
  _QWORD v125[2]; // [rsp+2B0h] [rbp-790h] BYREF
  unsigned __int64 v126[2]; // [rsp+2C0h] [rbp-780h] BYREF
  _QWORD v127[2]; // [rsp+2D0h] [rbp-770h] BYREF
  unsigned __int64 v128[2]; // [rsp+2E0h] [rbp-760h] BYREF
  char v129[16]; // [rsp+2F0h] [rbp-750h] BYREF
  unsigned __int64 v130[2]; // [rsp+300h] [rbp-740h] BYREF
  char v131[16]; // [rsp+310h] [rbp-730h] BYREF
  unsigned __int64 v132[2]; // [rsp+320h] [rbp-720h] BYREF
  char v133[16]; // [rsp+330h] [rbp-710h] BYREF
  unsigned __int64 v134[2]; // [rsp+340h] [rbp-700h] BYREF
  char v135[16]; // [rsp+350h] [rbp-6F0h] BYREF
  unsigned __int64 v136[2]; // [rsp+360h] [rbp-6E0h] BYREF
  __int64 v137; // [rsp+370h] [rbp-6D0h] BYREF
  char v138; // [rsp+378h] [rbp-6C8h]
  _BYTE v139[80]; // [rsp+380h] [rbp-6C0h] BYREF
  char v140[80]; // [rsp+3D0h] [rbp-670h] BYREF
  char v141[80]; // [rsp+420h] [rbp-620h] BYREF
  char v142[80]; // [rsp+470h] [rbp-5D0h] BYREF
  char v143[80]; // [rsp+4C0h] [rbp-580h] BYREF
  char v144[80]; // [rsp+510h] [rbp-530h] BYREF
  char v145[80]; // [rsp+560h] [rbp-4E0h] BYREF
  char v146[80]; // [rsp+5B0h] [rbp-490h] BYREF
  char v147[80]; // [rsp+600h] [rbp-440h] BYREF
  char v148[80]; // [rsp+650h] [rbp-3F0h] BYREF
  char v149[80]; // [rsp+6A0h] [rbp-3A0h] BYREF
  char v150[80]; // [rsp+6F0h] [rbp-350h] BYREF
  char v151[80]; // [rsp+740h] [rbp-300h] BYREF
  char v152[80]; // [rsp+790h] [rbp-2B0h] BYREF
  char v153[80]; // [rsp+7E0h] [rbp-260h] BYREF
  char v154[80]; // [rsp+830h] [rbp-210h] BYREF
  char v155[80]; // [rsp+880h] [rbp-1C0h] BYREF
  char v156[80]; // [rsp+8D0h] [rbp-170h] BYREF
  char v157[80]; // [rsp+920h] [rbp-120h] BYREF
  char v158[80]; // [rsp+970h] [rbp-D0h] BYREF
  _BYTE v159[80]; // [rsp+9C0h] [rbp-80h] BYREF
  _BYTE v160[48]; // [rsp+A10h] [rbp-30h] BYREF

  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)a1 = off_4A39A20;
  *(_DWORD *)(a1 + 16) = 1;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v96[0] = (unsigned __int64)v97;
  strcpy(v97, "mask");
  v96[1] = 4;
  v3 = sub_310D010();
  sub_310F6F0((__int64)v139, (__int64)v96, 0, v3, 8, (__int64)&qword_503F7F0);
  v99[0] = 0x656572665F7369LL;
  v98[0] = (unsigned __int64)v99;
  v98[1] = 7;
  v4 = sub_310D010();
  sub_310F6F0((__int64)v140, (__int64)v98, 0, v4, 8, (__int64)&qword_503F7F0);
  v100[0] = (unsigned __int64)v101;
  strcpy(v101, "nr_urgent");
  v100[1] = 9;
  v5 = sub_310D000();
  sub_310F6F0((__int64)v141, (__int64)v100, 0, v5, 4, (__int64)&qword_503F7F0);
  v102[0] = (unsigned __int64)v103;
  strcpy(v103, "nr_broken_hints");
  v102[1] = 15;
  v6 = sub_310D000();
  sub_310F6F0((__int64)v142, (__int64)v102, 0, v6, 4, (__int64)&qword_503F7F0);
  v104[0] = (unsigned __int64)v105;
  v105[0] = 0x746E69685F7369LL;
  v104[1] = 7;
  v7 = sub_310D010();
  sub_310F6F0((__int64)v143, (__int64)v104, 0, v7, 8, (__int64)&qword_503F7F0);
  v106[1] = 8;
  v106[0] = (unsigned __int64)v107;
  strcpy(v107, "is_local");
  v8 = sub_310D010();
  sub_310F6F0((__int64)v144, (__int64)v106, 0, v8, 8, (__int64)&qword_503F7F0);
  v136[0] = 19;
  v108[0] = (unsigned __int64)v109;
  v9 = (__m128i *)sub_22409D0((__int64)v108, v136, 0);
  v108[0] = (unsigned __int64)v9;
  v109[0] = v136[0];
  *v9 = _mm_load_si128((const __m128i *)&xmmword_44E60C0);
  v10 = v108[0];
  v9[1].m128i_i16[0] = 27746;
  v9[1].m128i_i8[2] = 101;
  v108[1] = v136[0];
  *(_BYTE *)(v10 + v136[0]) = 0;
  v11 = sub_310D000();
  sub_310F6F0((__int64)v145, (__int64)v108, 0, v11, 4, (__int64)&qword_503F7F0);
  v136[0] = 16;
  v110[0] = (unsigned __int64)v111;
  v110[0] = sub_22409D0((__int64)v110, v136, 0);
  v111[0] = v136[0];
  *(__m128i *)v110[0] = _mm_load_si128((const __m128i *)&xmmword_44E60D0);
  v110[1] = v136[0];
  *(_BYTE *)(v110[0] + v136[0]) = 0;
  v12 = sub_310D000();
  sub_310F6F0((__int64)v146, (__int64)v110, 0, v12, 4, (__int64)&qword_503F7F0);
  v136[0] = 20;
  v112[0] = (unsigned __int64)v113;
  v13 = (__m128i *)sub_22409D0((__int64)v112, v136, 0);
  v112[0] = (unsigned __int64)v13;
  v113[0] = v136[0];
  *v13 = _mm_load_si128((const __m128i *)&xmmword_44E60E0);
  v14 = v112[0];
  v13[1].m128i_i32[0] = 2019650911;
  v112[1] = v136[0];
  *(_BYTE *)(v14 + v136[0]) = 0;
  v15 = sub_310D000();
  sub_310F6F0((__int64)v147, (__int64)v112, 0, v15, 4, (__int64)&qword_503F7F0);
  v136[0] = 21;
  v114[0] = (unsigned __int64)v115;
  v16 = (__m128i *)sub_22409D0((__int64)v114, v136, 0);
  v114[0] = (unsigned __int64)v16;
  v115[0] = v136[0];
  *v16 = _mm_load_si128((const __m128i *)&xmmword_44E60F0);
  v17 = v114[0];
  v16[1].m128i_i32[0] = 1634557817;
  v16[1].m128i_i8[4] = 120;
  v114[1] = v136[0];
  *(_BYTE *)(v17 + v136[0]) = 0;
  v18 = sub_310D000();
  sub_310F6F0((__int64)v148, (__int64)v114, 0, v18, 4, (__int64)&qword_503F7F0);
  v136[0] = 26;
  v116[0] = (unsigned __int64)v117;
  v19 = (__m128i *)sub_22409D0((__int64)v116, v136, 0);
  v116[0] = (unsigned __int64)v19;
  v117[0] = v136[0];
  *v19 = _mm_load_si128((const __m128i *)&xmmword_44E6100);
  v20 = v116[0];
  qmemcpy(&v19[1], "tes_by_max", 10);
  v116[1] = v136[0];
  *(_BYTE *)(v20 + v136[0]) = 0;
  v21 = sub_310D000();
  sub_310F6F0((__int64)v149, (__int64)v116, 0, v21, 4, (__int64)&qword_503F7F0);
  v136[0] = 22;
  v118[0] = (unsigned __int64)v119;
  v22 = (__m128i *)sub_22409D0((__int64)v118, v136, 0);
  v118[0] = (unsigned __int64)v22;
  v119[0] = v136[0];
  *v22 = _mm_load_si128((const __m128i *)&xmmword_44E6110);
  v23 = v118[0];
  v22[1].m128i_i16[2] = 30817;
  v22[1].m128i_i32[0] = 1834973538;
  v118[1] = v136[0];
  *(_BYTE *)(v23 + v136[0]) = 0;
  v24 = sub_310D000();
  sub_310F6F0((__int64)v150, (__int64)v118, 0, v24, 4, (__int64)&qword_503F7F0);
  v136[0] = 19;
  v120[0] = (unsigned __int64)v121;
  v25 = (__m128i *)sub_22409D0((__int64)v120, v136, 0);
  v120[0] = (unsigned __int64)v25;
  v121[0] = v136[0];
  *v25 = _mm_load_si128((const __m128i *)&xmmword_44E6120);
  v26 = v120[0];
  v25[1].m128i_i16[0] = 24941;
  v25[1].m128i_i8[2] = 120;
  v120[1] = v136[0];
  *(_BYTE *)(v26 + v136[0]) = 0;
  v27 = sub_310D000();
  sub_310F6F0((__int64)v151, (__int64)v120, 0, v27, 4, (__int64)&qword_503F7F0);
  v136[0] = 20;
  v122[0] = (unsigned __int64)v123;
  v28 = (__m128i *)sub_22409D0((__int64)v122, v136, 0);
  v122[0] = (unsigned __int64)v28;
  v123[0] = v136[0];
  *v28 = _mm_load_si128((const __m128i *)&xmmword_44E6130);
  v29 = v122[0];
  v28[1].m128i_i32[0] = 2019650911;
  v122[1] = v136[0];
  *(_BYTE *)(v29 + v136[0]) = 0;
  v30 = sub_310D000();
  sub_310F6F0((__int64)v152, (__int64)v122, 0, v30, 4, (__int64)&qword_503F7F0);
  v136[0] = 18;
  v124[0] = (unsigned __int64)v125;
  v31 = sub_22409D0((__int64)v124, v136, 0);
  si128 = _mm_load_si128((const __m128i *)&xmmword_44E6140);
  v124[0] = v31;
  v125[0] = v136[0];
  *(_WORD *)(v31 + 16) = 30817;
  *(__m128i *)v31 = si128;
  v124[1] = v136[0];
  *(_BYTE *)(v124[0] + v136[0]) = 0;
  v33 = sub_310D000();
  sub_310F6F0((__int64)v153, (__int64)v124, 0, v33, 4, (__int64)&qword_503F7F0);
  v136[0] = 22;
  v126[0] = (unsigned __int64)v127;
  v34 = (__m128i *)sub_22409D0((__int64)v126, v136, 0);
  v126[0] = (unsigned __int64)v34;
  v127[0] = v136[0];
  *v34 = _mm_load_si128((const __m128i *)&xmmword_44E6150);
  v35 = v126[0];
  v34[1].m128i_i16[2] = 30817;
  v34[1].m128i_i32[0] = 1834973538;
  v126[1] = v136[0];
  *(_BYTE *)(v35 + v136[0]) = 0;
  v36 = sub_310D000();
  sub_310F6F0((__int64)v154, (__int64)v126, 0, v36, 4, (__int64)&qword_503F7F0);
  v128[0] = (unsigned __int64)v129;
  strcpy(v129, "liverange_size");
  v128[1] = 14;
  v37 = sub_310D000();
  sub_310F6F0((__int64)v155, (__int64)v128, 0, v37, 4, (__int64)&qword_503F7F0);
  v130[0] = (unsigned __int64)v131;
  strcpy(v131, "use_def_density");
  v130[1] = 15;
  v38 = sub_310D000();
  sub_310F6F0((__int64)v156, (__int64)v130, 0, v38, 4, (__int64)&qword_503F7F0);
  v132[0] = (unsigned __int64)v133;
  strcpy(v133, "max_stage");
  v132[1] = 9;
  v39 = sub_310D010();
  sub_310F6F0((__int64)v157, (__int64)v132, 0, v39, 8, (__int64)&qword_503F7F0);
  v134[0] = (unsigned __int64)v135;
  strcpy(v135, "min_stage");
  v134[1] = 9;
  v40 = sub_310D010();
  sub_310F6F0((__int64)v158, (__int64)v134, 0, v40, 8, (__int64)&qword_503F7F0);
  v41 = (_QWORD *)sub_22077B0(8u);
  v138 = 0;
  *v41 = 1;
  v95[0] = (unsigned __int64)v41;
  v136[0] = (unsigned __int64)&v137;
  v95[2] = (unsigned __int64)(v41 + 1);
  v95[1] = (unsigned __int64)(v41 + 1);
  v137 = 0x73736572676F7270LL;
  v136[1] = 8;
  v42 = sub_310D000();
  sub_310F6F0((__int64)v159, (__int64)v136, 0, v42, 4, (__int64)v95);
  v43 = *(_QWORD *)(a1 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 40) - v43) > 0x68F )
  {
    v59 = *(_QWORD *)(a1 + 32);
    v60 = v59 - v43;
    if ( (unsigned __int64)(v59 - v43) <= 0x68F )
    {
      v75 = &v139[v60];
      v76 = 0xCCCCCCCCCCCCCCCDLL * (v60 >> 4);
      if ( v60 )
      {
        v77 = v139;
        v78 = *(unsigned __int64 **)(a1 + 24);
        v79 = &v139[v60];
        do
        {
          sub_2240AE0(v78, (unsigned __int64 *)v77);
          v80 = *((_DWORD *)v77 + 8);
          v81 = (char **)(v77 + 40);
          v82 = (__int64)(v78 + 5);
          v77 += 80;
          v78 += 10;
          *((_DWORD *)v78 - 12) = v80;
          *((_DWORD *)v78 - 11) = *((_DWORD *)v77 - 11);
          sub_3592B50(v82, v81);
          *(v78 - 2) = *((_QWORD *)v77 - 2);
          *(v78 - 1) = *((_QWORD *)v77 - 1);
          --v76;
        }
        while ( v76 );
        v59 = *(_QWORD *)(a1 + 32);
        v75 = v79;
      }
      if ( v75 != v160 )
      {
        v83 = v75;
        do
        {
          if ( v59 )
          {
            v89 = *(_BYTE **)v83;
            v90 = *((_QWORD *)v83 + 1);
            v54 = v59;
            *(_QWORD *)v59 = v59 + 16;
            sub_3592E00((__int64 *)v59, v89, (__int64)&v89[v90]);
            v91 = *((_DWORD *)v83 + 8);
            v47 = (_BYTE *)*((_QWORD *)v83 + 5);
            *(_QWORD *)(v59 + 40) = 0;
            *(_QWORD *)(v59 + 48) = 0;
            *(_DWORD *)(v59 + 32) = v91;
            v92 = *((_DWORD *)v83 + 9);
            *(_QWORD *)(v59 + 56) = 0;
            *(_DWORD *)(v59 + 36) = v92;
            v86 = (_BYTE *)*((_QWORD *)v83 + 6);
            v93 = v86 - v47;
            if ( v86 == v47 )
            {
              v87 = 0;
              v93 = 0;
              v85 = 0;
            }
            else
            {
              if ( v93 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_101:
                sub_4261EA(v54, v47, v55);
              v84 = sub_22077B0(v93);
              v47 = (_BYTE *)*((_QWORD *)v83 + 5);
              v85 = (char *)v84;
              v86 = (_BYTE *)*((_QWORD *)v83 + 6);
              v87 = v86 - v47;
            }
            *(_QWORD *)(v59 + 40) = v85;
            *(_QWORD *)(v59 + 48) = v85;
            *(_QWORD *)(v59 + 56) = &v85[v93];
            if ( v47 != v86 )
              v85 = (char *)memmove(v85, v47, v87);
            v88 = *((_QWORD *)v83 + 8);
            *(_QWORD *)(v59 + 48) = &v85[v87];
            *(_QWORD *)(v59 + 64) = v88;
            *(_QWORD *)(v59 + 72) = *((_QWORD *)v83 + 9);
          }
          v83 += 80;
          v59 += 80;
        }
        while ( v83 != v160 );
      }
      *(_QWORD *)(a1 + 32) = v59;
    }
    else
    {
      v61 = *(unsigned __int64 **)(a1 + 24);
      v62 = v139;
      do
      {
        sub_2240AE0(v61, (unsigned __int64 *)v62);
        v63 = *((_DWORD *)v62 + 8);
        v64 = (char **)(v62 + 40);
        v65 = (__int64)(v61 + 5);
        v62 += 80;
        v61 += 10;
        *((_DWORD *)v61 - 12) = v63;
        *((_DWORD *)v61 - 11) = *((_DWORD *)v62 - 11);
        sub_3592B50(v65, v64);
        *(v61 - 2) = *((_QWORD *)v62 - 2);
        *(v61 - 1) = *((_QWORD *)v62 - 1);
      }
      while ( v62 != v160 );
      v66 = *(unsigned __int64 **)(a1 + 32);
      v67 = (unsigned __int64 *)(v43 + 1680);
      if ( v66 != v67 )
      {
        v68 = v67;
        do
        {
          v69 = v68[5];
          if ( v69 )
            j_j___libc_free_0(v69);
          if ( (unsigned __int64 *)*v68 != v68 + 2 )
            j_j___libc_free_0(*v68);
          v68 += 10;
        }
        while ( v66 != v68 );
        *(_QWORD *)(a1 + 32) = v67;
      }
    }
  }
  else
  {
    v44 = v139;
    v94 = sub_22077B0(0x690u);
    v45 = v94;
    do
    {
      if ( v45 )
      {
        v52 = *(_BYTE **)v44;
        v53 = *((_QWORD *)v44 + 1);
        v54 = v45;
        *(_QWORD *)v45 = v45 + 16;
        sub_3592E00((__int64 *)v45, v52, (__int64)&v52[v53]);
        v56 = *((_DWORD *)v44 + 8);
        v47 = (_BYTE *)*((_QWORD *)v44 + 5);
        *(_QWORD *)(v45 + 40) = 0;
        *(_QWORD *)(v45 + 48) = 0;
        *(_DWORD *)(v45 + 32) = v56;
        v57 = *((_DWORD *)v44 + 9);
        *(_QWORD *)(v45 + 56) = 0;
        *(_DWORD *)(v45 + 36) = v57;
        v49 = (_BYTE *)*((_QWORD *)v44 + 6);
        v58 = v49 - v47;
        if ( v49 == v47 )
        {
          v50 = 0;
          v58 = 0;
          v48 = 0;
        }
        else
        {
          if ( v58 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_101;
          v46 = sub_22077B0(v58);
          v47 = (_BYTE *)*((_QWORD *)v44 + 5);
          v48 = (char *)v46;
          v49 = (_BYTE *)*((_QWORD *)v44 + 6);
          v50 = v49 - v47;
        }
        *(_QWORD *)(v45 + 40) = v48;
        *(_QWORD *)(v45 + 48) = v48;
        *(_QWORD *)(v45 + 56) = &v48[v58];
        if ( v49 != v47 )
          v48 = (char *)memmove(v48, v47, v50);
        v51 = *((_QWORD *)v44 + 8);
        *(_QWORD *)(v45 + 48) = &v48[v50];
        *(_QWORD *)(v45 + 64) = v51;
        *(_QWORD *)(v45 + 72) = *((_QWORD *)v44 + 9);
      }
      v44 += 80;
      v45 += 80;
    }
    while ( v44 != v160 );
    v72 = *(unsigned __int64 **)(a1 + 32);
    v73 = *(unsigned __int64 **)(a1 + 24);
    if ( v72 != v73 )
    {
      do
      {
        v74 = v73[5];
        if ( v74 )
          j_j___libc_free_0(v74);
        if ( (unsigned __int64 *)*v73 != v73 + 2 )
          j_j___libc_free_0(*v73);
        v73 += 10;
      }
      while ( v72 != v73 );
      v73 = *(unsigned __int64 **)(a1 + 24);
    }
    if ( v73 )
      j_j___libc_free_0((unsigned __int64)v73);
    *(_QWORD *)(a1 + 24) = v94;
    *(_QWORD *)(a1 + 32) = v94 + 1680;
    *(_QWORD *)(a1 + 40) = v94 + 1680;
  }
  for ( i = (unsigned __int64 *)v159; ; i -= 10 )
  {
    v71 = i[5];
    if ( v71 )
      j_j___libc_free_0(v71);
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
    if ( v139 == (_BYTE *)i )
      break;
  }
  if ( (__int64 *)v136[0] != &v137 )
    j_j___libc_free_0(v136[0]);
  if ( v95[0] )
    j_j___libc_free_0(v95[0]);
  if ( (char *)v134[0] != v135 )
    j_j___libc_free_0(v134[0]);
  if ( (char *)v132[0] != v133 )
    j_j___libc_free_0(v132[0]);
  if ( (char *)v130[0] != v131 )
    j_j___libc_free_0(v130[0]);
  if ( (char *)v128[0] != v129 )
    j_j___libc_free_0(v128[0]);
  if ( (_QWORD *)v126[0] != v127 )
    j_j___libc_free_0(v126[0]);
  if ( (_QWORD *)v124[0] != v125 )
    j_j___libc_free_0(v124[0]);
  if ( (_QWORD *)v122[0] != v123 )
    j_j___libc_free_0(v122[0]);
  if ( (_QWORD *)v120[0] != v121 )
    j_j___libc_free_0(v120[0]);
  if ( (_QWORD *)v118[0] != v119 )
    j_j___libc_free_0(v118[0]);
  if ( (_QWORD *)v116[0] != v117 )
    j_j___libc_free_0(v116[0]);
  if ( (_QWORD *)v114[0] != v115 )
    j_j___libc_free_0(v114[0]);
  if ( (_QWORD *)v112[0] != v113 )
    j_j___libc_free_0(v112[0]);
  if ( (_QWORD *)v110[0] != v111 )
    j_j___libc_free_0(v110[0]);
  if ( (_QWORD *)v108[0] != v109 )
    j_j___libc_free_0(v108[0]);
  if ( (char *)v106[0] != v107 )
    j_j___libc_free_0(v106[0]);
  if ( (_QWORD *)v104[0] != v105 )
    j_j___libc_free_0(v104[0]);
  if ( (char *)v102[0] != v103 )
    j_j___libc_free_0(v102[0]);
  if ( (char *)v100[0] != v101 )
    j_j___libc_free_0(v100[0]);
  if ( (_QWORD *)v98[0] != v99 )
    j_j___libc_free_0(v98[0]);
  if ( (char *)v96[0] != v97 )
    j_j___libc_free_0(v96[0]);
}
