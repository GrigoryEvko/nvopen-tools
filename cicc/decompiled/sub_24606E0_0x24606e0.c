// Function: sub_24606E0
// Address: 0x24606e0
//
__int64 __fastcall sub_24606E0(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, _QWORD *a5)
{
  _QWORD *i; // rbx
  unsigned __int64 v6; // r13
  _BYTE *v7; // rbx
  _QWORD *v8; // r14
  char v9; // si
  __int64 v10; // rax
  __int64 v12; // rax
  char v13; // r12
  _QWORD *v14; // r14
  _QWORD *v15; // r13
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // r13d
  _QWORD *v27; // r15
  _QWORD *v28; // r14
  unsigned __int64 v29; // rsi
  _QWORD *v30; // rax
  _QWORD *v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // rdx
  _QWORD *v40; // rdi
  double v41; // r14
  unsigned __int64 v42; // rax
  double v43; // xmm0_8
  double v44; // xmm0_8
  char v45; // r14
  _QWORD *v46; // rax
  __int64 v47; // rdx
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  _QWORD *v52; // rax
  __int64 v53; // r12
  __int64 v54; // rdi
  __int64 v55; // rax
  char *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r12
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r9
  __m128i v63; // xmm4
  __int64 v64; // rdx
  __m128i v65; // xmm6
  unsigned __int64 *v66; // r13
  unsigned __int64 *v67; // r14
  unsigned __int64 v68; // rdi
  unsigned __int64 *v69; // r12
  unsigned __int64 *v70; // r13
  unsigned __int64 v71; // rdi
  __int64 v72; // rax
  char *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r12
  __int64 v76; // r12
  __int64 v77; // rax
  __int64 v78; // r8
  __int64 v79; // r9
  __m128i v80; // xmm3
  __int64 v81; // rdx
  __m128i v82; // xmm5
  unsigned __int64 *v83; // r13
  unsigned __int64 *v84; // r14
  unsigned __int64 v85; // rdi
  unsigned __int64 *v86; // r12
  unsigned __int64 v87; // rdi
  __int64 v88; // rax
  unsigned __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // r15
  const char *v95; // rax
  size_t v96; // rdx
  __int64 v101; // [rsp+38h] [rbp-6D8h]
  __int64 v102; // [rsp+38h] [rbp-6D8h]
  __int64 v103; // [rsp+40h] [rbp-6D0h]
  _QWORD *v105; // [rsp+68h] [rbp-6A8h]
  __int64 v106; // [rsp+70h] [rbp-6A0h]
  unsigned __int64 v107; // [rsp+88h] [rbp-688h]
  __int64 v108[2]; // [rsp+90h] [rbp-680h] BYREF
  _QWORD v109[2]; // [rsp+A0h] [rbp-670h] BYREF
  __int64 v110[2]; // [rsp+B0h] [rbp-660h] BYREF
  _QWORD v111[2]; // [rsp+C0h] [rbp-650h] BYREF
  __m128i v112; // [rsp+D0h] [rbp-640h]
  __int64 v113[2]; // [rsp+E0h] [rbp-630h] BYREF
  _QWORD v114[2]; // [rsp+F0h] [rbp-620h] BYREF
  __int64 v115[2]; // [rsp+100h] [rbp-610h] BYREF
  _QWORD v116[2]; // [rsp+110h] [rbp-600h] BYREF
  __m128i v117; // [rsp+120h] [rbp-5F0h]
  __int64 v118[2]; // [rsp+130h] [rbp-5E0h] BYREF
  _QWORD v119[2]; // [rsp+140h] [rbp-5D0h] BYREF
  __int64 v120[2]; // [rsp+150h] [rbp-5C0h] BYREF
  _QWORD v121[2]; // [rsp+160h] [rbp-5B0h] BYREF
  __m128i v122; // [rsp+170h] [rbp-5A0h]
  _BYTE *v123; // [rsp+180h] [rbp-590h] BYREF
  __int64 v124; // [rsp+188h] [rbp-588h]
  _QWORD v125[2]; // [rsp+190h] [rbp-580h] BYREF
  _BYTE *v126; // [rsp+1A0h] [rbp-570h]
  __int64 v127; // [rsp+1A8h] [rbp-568h]
  _QWORD v128[2]; // [rsp+1B0h] [rbp-560h] BYREF
  __m128i v129; // [rsp+1C0h] [rbp-550h] BYREF
  _BYTE *v130; // [rsp+1D0h] [rbp-540h] BYREF
  __int64 v131; // [rsp+1D8h] [rbp-538h]
  _QWORD v132[2]; // [rsp+1E0h] [rbp-530h] BYREF
  _BYTE *v133; // [rsp+1F0h] [rbp-520h]
  __int64 v134; // [rsp+1F8h] [rbp-518h]
  _QWORD v135[2]; // [rsp+200h] [rbp-510h] BYREF
  __m128i v136; // [rsp+210h] [rbp-500h] BYREF
  _BYTE *v137; // [rsp+220h] [rbp-4F0h] BYREF
  __int64 v138; // [rsp+228h] [rbp-4E8h]
  _QWORD v139[2]; // [rsp+230h] [rbp-4E0h] BYREF
  _BYTE *v140; // [rsp+240h] [rbp-4D0h]
  __int64 v141; // [rsp+248h] [rbp-4C8h]
  _QWORD v142[2]; // [rsp+250h] [rbp-4C0h] BYREF
  __m128i v143; // [rsp+260h] [rbp-4B0h] BYREF
  _BYTE *v144; // [rsp+270h] [rbp-4A0h] BYREF
  __int64 v145; // [rsp+278h] [rbp-498h]
  _BYTE v146[256]; // [rsp+280h] [rbp-490h] BYREF
  void *v147; // [rsp+380h] [rbp-390h] BYREF
  int v148; // [rsp+388h] [rbp-388h]
  char v149; // [rsp+38Ch] [rbp-384h]
  __int64 v150; // [rsp+390h] [rbp-380h]
  __m128i v151; // [rsp+398h] [rbp-378h]
  __int64 v152; // [rsp+3A8h] [rbp-368h]
  __m128i v153; // [rsp+3B0h] [rbp-360h]
  __m128i v154; // [rsp+3C0h] [rbp-350h]
  unsigned __int64 *v155; // [rsp+3D0h] [rbp-340h] BYREF
  __int64 v156; // [rsp+3D8h] [rbp-338h]
  _BYTE v157[320]; // [rsp+3E0h] [rbp-330h] BYREF
  char v158; // [rsp+520h] [rbp-1F0h]
  int v159; // [rsp+524h] [rbp-1ECh]
  __int64 v160; // [rsp+528h] [rbp-1E8h]
  _QWORD *v161; // [rsp+530h] [rbp-1E0h] BYREF
  __int64 v162; // [rsp+538h] [rbp-1D8h]
  unsigned __int64 *v163; // [rsp+580h] [rbp-190h]
  unsigned int v164; // [rsp+588h] [rbp-188h]
  _BYTE v165[384]; // [rsp+590h] [rbp-180h] BYREF

  v144 = v146;
  v145 = 0x1000000000LL;
  v107 = 0;
  v105 = *(_QWORD **)(a1 + 80);
  v103 = a1 + 72;
  if ( v105 == (_QWORD *)(a1 + 72) )
  {
    LODWORD(v6) = 0;
  }
  else
  {
    do
    {
      if ( !v105 )
        BUG();
      for ( i = (_QWORD *)v105[4]; v105 + 3 != i; i = (_QWORD *)i[1] )
      {
        if ( !i )
          BUG();
        if ( *((_BYTE *)i - 24) != 85 )
          continue;
        v12 = *(i - 7);
        if ( !v12 )
          continue;
        v13 = *(_BYTE *)v12;
        if ( *(_BYTE *)v12
          || *(_QWORD *)(v12 + 24) != i[7]
          || (*(_BYTE *)(v12 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v12 + 36) - 5) > 1 )
        {
          continue;
        }
        v14 = sub_C52410();
        v15 = v14 + 1;
        v16 = sub_C959E0();
        v17 = (_QWORD *)v14[2];
        if ( v17 )
        {
          v18 = v14 + 1;
          do
          {
            while ( 1 )
            {
              v19 = v17[2];
              v20 = v17[3];
              if ( v16 <= v17[4] )
                break;
              v17 = (_QWORD *)v17[3];
              if ( !v20 )
                goto LABEL_26;
            }
            v18 = v17;
            v17 = (_QWORD *)v17[2];
          }
          while ( v19 );
LABEL_26:
          if ( v15 != v18 && v16 >= v18[4] )
            v15 = v18;
        }
        if ( v15 == (_QWORD *)((char *)sub_C52410() + 8) )
          goto LABEL_37;
        v21 = v15[7];
        if ( !v21 )
          goto LABEL_37;
        v22 = v15 + 6;
        do
        {
          while ( 1 )
          {
            v23 = *(_QWORD *)(v21 + 16);
            v24 = *(_QWORD *)(v21 + 24);
            if ( *(_DWORD *)(v21 + 32) >= dword_4FE7808 )
              break;
            v21 = *(_QWORD *)(v21 + 24);
            if ( !v24 )
              goto LABEL_35;
          }
          v22 = (_QWORD *)v21;
          v21 = *(_QWORD *)(v21 + 16);
        }
        while ( v23 );
LABEL_35:
        if ( v15 + 6 == v22 || dword_4FE7808 < *((_DWORD *)v22 + 8) || (v26 = qword_4FE7888, !*((_DWORD *)v22 + 9)) )
        {
LABEL_37:
          v25 = *(i - 7);
          if ( !v25 || *(_BYTE *)v25 || *(_QWORD *)(v25 + 24) != i[7] )
            BUG();
          v26 = 0;
          if ( *(_DWORD *)(v25 + 36) == 6 )
          {
            v88 = i[-4 * (*((_DWORD *)i - 5) & 0x7FFFFFF) - 3];
            v89 = *(_QWORD *)(v88 + 24);
            if ( *(_DWORD *)(v88 + 32) > 0x40u )
              v89 = *(_QWORD *)v89;
            v26 = 0;
            if ( (__int64)(a5[1] - *a5) >> 2 > v89 )
              v26 = *(_DWORD *)(*a5 + 4 * v89);
          }
        }
        v27 = sub_C52410();
        v28 = v27 + 1;
        v29 = sub_C959E0();
        v30 = (_QWORD *)v27[2];
        if ( v30 )
        {
          v31 = v27 + 1;
          do
          {
            while ( 1 )
            {
              v32 = v30[2];
              v33 = v30[3];
              if ( v29 <= v30[4] )
                break;
              v30 = (_QWORD *)v30[3];
              if ( !v33 )
                goto LABEL_46;
            }
            v31 = v30;
            v30 = (_QWORD *)v30[2];
          }
          while ( v32 );
LABEL_46:
          if ( v28 != v31 && v29 >= v31[4] )
            v28 = v31;
        }
        if ( v28 != (_QWORD *)((char *)sub_C52410() + 8) )
        {
          v36 = v28[7];
          v34 = (__int64)(v28 + 6);
          if ( v36 )
          {
            v37 = v28 + 6;
            do
            {
              while ( 1 )
              {
                v38 = *(_QWORD *)(v36 + 16);
                v39 = *(_QWORD *)(v36 + 24);
                if ( *(_DWORD *)(v36 + 32) >= dword_4FE7728 )
                  break;
                v36 = *(_QWORD *)(v36 + 24);
                if ( !v39 )
                  goto LABEL_55;
              }
              v37 = (_QWORD *)v36;
              v36 = *(_QWORD *)(v36 + 16);
            }
            while ( v38 );
LABEL_55:
            if ( (_QWORD *)v34 != v37 && dword_4FE7728 >= *((_DWORD *)v37 + 8) && *((_DWORD *)v37 + 9) )
            {
              v40 = (_QWORD *)v107;
              v41 = *(float *)&qword_4FE77A8;
              if ( !v107 )
              {
                v94 = *(_QWORD *)(a1 + 40);
                v95 = sub_BD5D20(a1);
                sub_BA89D0((__int64 *)&v161, v94, v95, v96);
                v40 = v161;
                v107 = (unsigned __int64)v161;
              }
              v42 = sub_C88A20(v40);
              if ( (v42 & 0x8000000000000000LL) != 0LL )
                v43 = (double)(int)(v42 & 1 | (v42 >> 1)) + (double)(int)(v42 & 1 | (v42 >> 1));
              else
                v43 = (double)(int)v42;
              v44 = (v43 + 0.0) * 5.421010862427522e-20;
              if ( v44 >= 1.0 )
                v44 = 0.9999999999999999;
              if ( v41 <= v44 )
                goto LABEL_65;
            }
          }
        }
        if ( v26 == 1000000 )
          goto LABEL_65;
        if ( !a3 )
          goto LABEL_71;
        v46 = (_QWORD *)sub_FDD2C0(a2, i[2], 0);
        v162 = v47;
        v48 = 0;
        v161 = v46;
        if ( (_BYTE)v162 )
          v48 = (unsigned __int64)v161;
        if ( sub_D85370(a3, v26, v48) )
        {
LABEL_65:
          v13 = 1;
          v45 = 1;
        }
        else
        {
LABEL_71:
          v45 = 0;
        }
        v49 = v106;
        LOBYTE(v49) = v13;
        v106 = v49;
        v50 = (unsigned int)v145;
        v51 = (unsigned int)v145 + 1LL;
        if ( v51 > HIDWORD(v145) )
        {
          sub_C8D5F0((__int64)&v144, v146, v51, 0x10u, v34, v35);
          v50 = (unsigned int)v145;
        }
        v52 = &v144[16 * v50];
        *v52 = i - 3;
        v52[1] = v106;
        LODWORD(v145) = v145 + 1;
        v53 = *a4;
        v54 = *a4;
        if ( v45 )
        {
          v55 = sub_B2BE50(v54);
          if ( sub_B6EA50(v55)
            || (v92 = sub_B2BE50(v53),
                v93 = sub_B6F970(v92),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v93 + 48LL))(v93)) )
          {
            sub_B16080((__int64)&v123, "Kind", 4, (unsigned __int8 *)i[-4 * (*((_DWORD *)i - 5) & 0x7FFFFFF) - 3]);
            sub_B16080((__int64)&v130, "Function", 8, *(unsigned __int8 **)(i[2] + 72LL));
            v56 = (char *)sub_BD5D20(i[2]);
            sub_B16430((__int64)&v137, "Block", 5u, v56, v57);
            sub_B174A0((__int64)&v161, (__int64)"lower-allow-check", (__int64)"Removed", 7, (__int64)(i - 3));
            sub_B18290((__int64)&v161, "Removed check: Kind=", 0x14u);
            v118[0] = (__int64)v119;
            sub_2460190(v118, v123, (__int64)&v123[v124]);
            v120[0] = (__int64)v121;
            sub_2460190(v120, v126, (__int64)&v126[v127]);
            v122 = _mm_load_si128(&v129);
            v58 = sub_23FD640((__int64)&v161, (__int64)v118);
            sub_B18290(v58, " F=", 3u);
            v113[0] = (__int64)v114;
            sub_2460190(v113, v130, (__int64)&v130[v131]);
            v115[0] = (__int64)v116;
            sub_2460190(v115, v133, (__int64)&v133[v134]);
            v117 = _mm_load_si128(&v136);
            v59 = sub_23FD640(v58, (__int64)v113);
            sub_B18290(v59, " BB=", 4u);
            v108[0] = (__int64)v109;
            sub_2460190(v108, v137, (__int64)&v137[v138]);
            v110[0] = (__int64)v111;
            sub_2460190(v110, v140, (__int64)&v140[v141]);
            v112 = _mm_load_si128(&v143);
            v60 = sub_23FD640(v59, (__int64)v108);
            v148 = *(_DWORD *)(v60 + 8);
            v149 = *(_BYTE *)(v60 + 12);
            v150 = *(_QWORD *)(v60 + 16);
            v63 = _mm_loadu_si128((const __m128i *)(v60 + 24));
            v147 = &unk_49D9D40;
            v151 = v63;
            v64 = *(_QWORD *)(v60 + 40);
            v152 = v64;
            v153 = _mm_loadu_si128((const __m128i *)(v60 + 48));
            v65 = _mm_loadu_si128((const __m128i *)(v60 + 64));
            v155 = (unsigned __int64 *)v157;
            v156 = 0x400000000LL;
            v154 = v65;
            if ( *(_DWORD *)(v60 + 88) )
            {
              v101 = v60;
              sub_2460460((__int64)&v155, v60 + 80, v64, (__int64)v111, v61, v62);
              v60 = v101;
            }
            v158 = *(_BYTE *)(v60 + 416);
            v159 = *(_DWORD *)(v60 + 420);
            v160 = *(_QWORD *)(v60 + 424);
            v147 = &unk_49D9D78;
            if ( (_QWORD *)v110[0] != v111 )
              j_j___libc_free_0(v110[0]);
            if ( (_QWORD *)v108[0] != v109 )
              j_j___libc_free_0(v108[0]);
            if ( (_QWORD *)v115[0] != v116 )
              j_j___libc_free_0(v115[0]);
            if ( (_QWORD *)v113[0] != v114 )
              j_j___libc_free_0(v113[0]);
            if ( (_QWORD *)v120[0] != v121 )
              j_j___libc_free_0(v120[0]);
            if ( (_QWORD *)v118[0] != v119 )
              j_j___libc_free_0(v118[0]);
            v66 = v163;
            v161 = &unk_49D9D40;
            v67 = &v163[10 * v164];
            if ( v163 != v67 )
            {
              do
              {
                v67 -= 10;
                v68 = v67[4];
                if ( (unsigned __int64 *)v68 != v67 + 6 )
                  j_j___libc_free_0(v68);
                if ( (unsigned __int64 *)*v67 != v67 + 2 )
                  j_j___libc_free_0(*v67);
              }
              while ( v66 != v67 );
              v67 = v163;
            }
            if ( v67 != (unsigned __int64 *)v165 )
              _libc_free((unsigned __int64)v67);
            if ( v140 != (_BYTE *)v142 )
              j_j___libc_free_0((unsigned __int64)v140);
            if ( v137 != (_BYTE *)v139 )
              j_j___libc_free_0((unsigned __int64)v137);
            if ( v133 != (_BYTE *)v135 )
              j_j___libc_free_0((unsigned __int64)v133);
            if ( v130 != (_BYTE *)v132 )
              j_j___libc_free_0((unsigned __int64)v130);
            if ( v126 != (_BYTE *)v128 )
              j_j___libc_free_0((unsigned __int64)v126);
            if ( v123 != (_BYTE *)v125 )
              j_j___libc_free_0((unsigned __int64)v123);
            sub_1049740(a4, (__int64)&v147);
            v69 = v155;
            v147 = &unk_49D9D40;
            v70 = &v155[10 * (unsigned int)v156];
            if ( v155 != v70 )
            {
              do
              {
                v70 -= 10;
                v71 = v70[4];
                if ( (unsigned __int64 *)v71 != v70 + 6 )
                  j_j___libc_free_0(v71);
                if ( (unsigned __int64 *)*v70 != v70 + 2 )
                  j_j___libc_free_0(*v70);
              }
              while ( v69 != v70 );
              goto LABEL_117;
            }
            goto LABEL_118;
          }
        }
        else
        {
          v72 = sub_B2BE50(v54);
          if ( sub_B6EA50(v72)
            || (v90 = sub_B2BE50(v53),
                v91 = sub_B6F970(v90),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v91 + 48LL))(v91)) )
          {
            sub_B16080((__int64)&v123, "Kind", 4, (unsigned __int8 *)i[-4 * (*((_DWORD *)i - 5) & 0x7FFFFFF) - 3]);
            sub_B16080((__int64)&v130, "Function", 8, *(unsigned __int8 **)(i[2] + 72LL));
            v73 = (char *)sub_BD5D20(i[2]);
            sub_B16430((__int64)&v137, "Block", 5u, v73, v74);
            sub_B176B0((__int64)&v161, (__int64)"lower-allow-check", (__int64)"Allowed", 7, (__int64)(i - 3));
            sub_B18290((__int64)&v161, "Allowed check: Kind=", 0x14u);
            v118[0] = (__int64)v119;
            sub_2460190(v118, v123, (__int64)&v123[v124]);
            v120[0] = (__int64)v121;
            sub_2460190(v120, v126, (__int64)&v126[v127]);
            v122 = _mm_load_si128(&v129);
            v75 = sub_2445430((__int64)&v161, (__int64)v118);
            sub_B18290(v75, " F=", 3u);
            v113[0] = (__int64)v114;
            sub_2460190(v113, v130, (__int64)&v130[v131]);
            v115[0] = (__int64)v116;
            sub_2460190(v115, v133, (__int64)&v133[v134]);
            v117 = _mm_load_si128(&v136);
            v76 = sub_2445430(v75, (__int64)v113);
            sub_B18290(v76, " BB=", 4u);
            v108[0] = (__int64)v109;
            sub_2460190(v108, v137, (__int64)&v137[v138]);
            v110[0] = (__int64)v111;
            sub_2460190(v110, v140, (__int64)&v140[v141]);
            v112 = _mm_load_si128(&v143);
            v77 = sub_2445430(v76, (__int64)v108);
            v148 = *(_DWORD *)(v77 + 8);
            v149 = *(_BYTE *)(v77 + 12);
            v150 = *(_QWORD *)(v77 + 16);
            v80 = _mm_loadu_si128((const __m128i *)(v77 + 24));
            v147 = &unk_49D9D40;
            v151 = v80;
            v81 = *(_QWORD *)(v77 + 40);
            v152 = v81;
            v153 = _mm_loadu_si128((const __m128i *)(v77 + 48));
            v82 = _mm_loadu_si128((const __m128i *)(v77 + 64));
            v155 = (unsigned __int64 *)v157;
            v156 = 0x400000000LL;
            v154 = v82;
            if ( *(_DWORD *)(v77 + 88) )
            {
              v102 = v77;
              sub_2460460((__int64)&v155, v77 + 80, v81, (__int64)v111, v78, v79);
              v77 = v102;
            }
            v158 = *(_BYTE *)(v77 + 416);
            v159 = *(_DWORD *)(v77 + 420);
            v160 = *(_QWORD *)(v77 + 424);
            v147 = &unk_49D9DB0;
            if ( (_QWORD *)v110[0] != v111 )
              j_j___libc_free_0(v110[0]);
            if ( (_QWORD *)v108[0] != v109 )
              j_j___libc_free_0(v108[0]);
            if ( (_QWORD *)v115[0] != v116 )
              j_j___libc_free_0(v115[0]);
            if ( (_QWORD *)v113[0] != v114 )
              j_j___libc_free_0(v113[0]);
            if ( (_QWORD *)v120[0] != v121 )
              j_j___libc_free_0(v120[0]);
            if ( (_QWORD *)v118[0] != v119 )
              j_j___libc_free_0(v118[0]);
            v83 = v163;
            v161 = &unk_49D9D40;
            v84 = &v163[10 * v164];
            if ( v163 != v84 )
            {
              do
              {
                v84 -= 10;
                v85 = v84[4];
                if ( (unsigned __int64 *)v85 != v84 + 6 )
                  j_j___libc_free_0(v85);
                if ( (unsigned __int64 *)*v84 != v84 + 2 )
                  j_j___libc_free_0(*v84);
              }
              while ( v83 != v84 );
              v84 = v163;
            }
            if ( v84 != (unsigned __int64 *)v165 )
              _libc_free((unsigned __int64)v84);
            if ( v140 != (_BYTE *)v142 )
              j_j___libc_free_0((unsigned __int64)v140);
            if ( v137 != (_BYTE *)v139 )
              j_j___libc_free_0((unsigned __int64)v137);
            if ( v133 != (_BYTE *)v135 )
              j_j___libc_free_0((unsigned __int64)v133);
            if ( v130 != (_BYTE *)v132 )
              j_j___libc_free_0((unsigned __int64)v130);
            if ( v126 != (_BYTE *)v128 )
              j_j___libc_free_0((unsigned __int64)v126);
            if ( v123 != (_BYTE *)v125 )
              j_j___libc_free_0((unsigned __int64)v123);
            sub_1049740(a4, (__int64)&v147);
            v86 = v155;
            v147 = &unk_49D9D40;
            v70 = &v155[10 * (unsigned int)v156];
            if ( v155 != v70 )
            {
              do
              {
                v70 -= 10;
                v87 = v70[4];
                if ( (unsigned __int64 *)v87 != v70 + 6 )
                  j_j___libc_free_0(v87);
                if ( (unsigned __int64 *)*v70 != v70 + 2 )
                  j_j___libc_free_0(*v70);
              }
              while ( v86 != v70 );
LABEL_117:
              v70 = v155;
            }
LABEL_118:
            if ( v70 != (unsigned __int64 *)v157 )
              _libc_free((unsigned __int64)v70);
          }
        }
      }
      v105 = (_QWORD *)v105[1];
    }
    while ( (_QWORD *)v103 != v105 );
    v6 = (unsigned __int64)v144;
    v7 = &v144[16 * (unsigned int)v145];
    if ( v7 == v144 )
    {
      LOBYTE(v6) = (_DWORD)v145 != 0;
    }
    else
    {
      do
      {
        v8 = *(_QWORD **)v6;
        v9 = *(_BYTE *)(v6 + 8);
        v6 += 16LL;
        v10 = sub_AD64A0(v8[1], v9 ^ 1u);
        sub_BD84D0((__int64)v8, v10);
        sub_B43D60(v8);
      }
      while ( v7 != (_BYTE *)v6 );
      LOBYTE(v6) = (_DWORD)v145 != 0;
    }
    if ( v107 )
      j_j___libc_free_0(v107);
  }
  if ( v144 != v146 )
    _libc_free((unsigned __int64)v144);
  return (unsigned int)v6;
}
