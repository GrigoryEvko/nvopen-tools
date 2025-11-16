// Function: sub_26ACCC0
// Address: 0x26accc0
//
_BOOL8 __fastcall sub_26ACCC0(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rsi
  char *v5; // rax
  char *v6; // rax
  __int64 *v7; // rax
  __int64 *v8; // rbx
  __int64 *v9; // r13
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // r14
  char *v21; // r15
  unsigned __int8 v22; // al
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 *v25; // rax
  unsigned __int64 v26; // rax
  __int16 v27; // ax
  int v28; // r13d
  unsigned __int64 v29; // r14
  __int64 v30; // r15
  __int64 v31; // r12
  __int64 v32; // rax
  signed __int64 v33; // rax
  int v34; // edx
  bool v35; // of
  __int64 *v36; // r13
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // rbx
  __int64 v43; // r8
  __int64 v44; // r9
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rcx
  unsigned __int64 *v51; // rbx
  __int64 v52; // r8
  unsigned __int64 *v53; // r14
  unsigned __int64 v54; // rdi
  unsigned __int64 *v55; // rbx
  unsigned __int64 *v56; // r13
  unsigned __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rbx
  __int64 v60; // r12
  __int64 v61; // rsi
  unsigned __int64 v62; // r14
  unsigned __int64 v63; // r12
  __int64 v64; // r15
  __int64 *v65; // rbx
  __int64 *v66; // r13
  __int64 v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rax
  unsigned int v70; // eax
  __int64 v71; // rdx
  char v72; // al
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // rdi
  __int64 *v75; // rbx
  __int64 *v76; // r12
  __int64 v77; // rsi
  __int64 v78; // rdi
  __int64 *v79; // rax
  __int64 *v80; // rbx
  __int64 *v81; // r14
  __int64 v82; // rdi
  unsigned int v83; // ecx
  __int64 v84; // rsi
  __int64 *v85; // rbx
  __int64 *v86; // r12
  __int64 v87; // rsi
  __int64 v88; // rdi
  _BYTE *v89; // rbx
  _BYTE *v90; // r12
  unsigned __int64 v91; // r13
  unsigned __int64 v92; // rdi
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // [rsp+48h] [rbp-938h]
  __int64 v96; // [rsp+80h] [rbp-900h]
  bool v97; // [rsp+88h] [rbp-8F8h]
  unsigned __int8 **v98; // [rsp+88h] [rbp-8F8h]
  __int64 v99; // [rsp+88h] [rbp-8F8h]
  __int64 v100; // [rsp+90h] [rbp-8F0h] BYREF
  __int64 v101; // [rsp+98h] [rbp-8E8h]
  __int64 v102; // [rsp+A0h] [rbp-8E0h]
  __int64 v103; // [rsp+A8h] [rbp-8D8h]
  __int64 *v104; // [rsp+B0h] [rbp-8D0h]
  __int64 v105; // [rsp+B8h] [rbp-8C8h]
  __int64 v106; // [rsp+C0h] [rbp-8C0h] BYREF
  __int64 v107; // [rsp+C8h] [rbp-8B8h]
  __int64 v108; // [rsp+D0h] [rbp-8B0h]
  __int64 v109; // [rsp+D8h] [rbp-8A8h]
  __int64 *v110; // [rsp+E0h] [rbp-8A0h]
  __int64 v111; // [rsp+E8h] [rbp-898h]
  __int64 v112; // [rsp+F0h] [rbp-890h] BYREF
  __int64 v113; // [rsp+F8h] [rbp-888h]
  __int64 v114; // [rsp+100h] [rbp-880h]
  __int64 v115; // [rsp+108h] [rbp-878h]
  unsigned __int64 *v116; // [rsp+110h] [rbp-870h]
  __int64 v117; // [rsp+118h] [rbp-868h]
  unsigned __int64 v118[2]; // [rsp+120h] [rbp-860h] BYREF
  __int64 v119; // [rsp+130h] [rbp-850h] BYREF
  __int64 *v120; // [rsp+140h] [rbp-840h]
  __int64 v121; // [rsp+150h] [rbp-830h] BYREF
  unsigned __int8 **v122; // [rsp+170h] [rbp-810h] BYREF
  __int64 v123; // [rsp+178h] [rbp-808h]
  _BYTE v124[64]; // [rsp+180h] [rbp-800h] BYREF
  __int64 v125; // [rsp+1C0h] [rbp-7C0h]
  __int64 v126; // [rsp+1C8h] [rbp-7B8h]
  __int64 v127; // [rsp+1D0h] [rbp-7B0h]
  unsigned __int64 v128[2]; // [rsp+1E0h] [rbp-7A0h] BYREF
  char v129; // [rsp+1F0h] [rbp-790h] BYREF
  _BYTE *v130; // [rsp+1F8h] [rbp-788h]
  __int64 v131; // [rsp+200h] [rbp-780h]
  _BYTE v132[56]; // [rsp+208h] [rbp-778h] BYREF
  __int64 v133; // [rsp+240h] [rbp-740h]
  __int64 v134; // [rsp+248h] [rbp-738h]
  char v135; // [rsp+250h] [rbp-730h]
  __int64 v136; // [rsp+254h] [rbp-72Ch]
  char v137[8]; // [rsp+260h] [rbp-720h] BYREF
  __int64 v138; // [rsp+268h] [rbp-718h]
  unsigned int v139; // [rsp+278h] [rbp-708h]
  unsigned __int64 v140; // [rsp+280h] [rbp-700h]
  unsigned __int64 v141; // [rsp+288h] [rbp-6F8h]
  __int64 v142; // [rsp+298h] [rbp-6E8h]
  __int64 i; // [rsp+2A0h] [rbp-6E0h]
  __int64 *v144; // [rsp+2A8h] [rbp-6D8h]
  unsigned int v145; // [rsp+2B0h] [rbp-6D0h]
  char v146; // [rsp+2B8h] [rbp-6C8h] BYREF
  __int64 *v147; // [rsp+2D8h] [rbp-6A8h]
  unsigned int v148; // [rsp+2E0h] [rbp-6A0h]
  __int64 v149; // [rsp+2E8h] [rbp-698h] BYREF
  unsigned __int64 v150[2]; // [rsp+300h] [rbp-680h] BYREF
  char v151; // [rsp+310h] [rbp-670h] BYREF
  __int64 v152; // [rsp+398h] [rbp-5E8h]
  unsigned int v153; // [rsp+3A8h] [rbp-5D8h]
  __int64 v154; // [rsp+3B8h] [rbp-5C8h]
  unsigned int v155; // [rsp+3C8h] [rbp-5B8h]
  _BYTE v156[64]; // [rsp+3D0h] [rbp-5B0h] BYREF
  __int64 v157; // [rsp+410h] [rbp-570h]
  unsigned int v158; // [rsp+420h] [rbp-560h]
  unsigned __int64 *v159; // [rsp+428h] [rbp-558h]
  char *v160; // [rsp+438h] [rbp-548h] BYREF
  char v161; // [rsp+448h] [rbp-538h] BYREF
  _QWORD *v162; // [rsp+478h] [rbp-508h]
  _QWORD v163[6]; // [rsp+488h] [rbp-4F8h] BYREF
  unsigned int v164; // [rsp+4B8h] [rbp-4C8h]
  _QWORD *v165; // [rsp+4C0h] [rbp-4C0h]
  _QWORD v166[13]; // [rsp+4D0h] [rbp-4B0h] BYREF
  char v167; // [rsp+538h] [rbp-448h] BYREF
  _QWORD v168[2]; // [rsp+578h] [rbp-408h] BYREF
  char v169; // [rsp+588h] [rbp-3F8h] BYREF
  char v170; // [rsp+5E8h] [rbp-398h] BYREF
  void *v171; // [rsp+5F0h] [rbp-390h] BYREF
  int v172; // [rsp+5F8h] [rbp-388h]
  char v173; // [rsp+5FCh] [rbp-384h]
  __int64 v174; // [rsp+600h] [rbp-380h]
  __m128i v175; // [rsp+608h] [rbp-378h]
  __int64 v176; // [rsp+618h] [rbp-368h]
  __m128i v177; // [rsp+620h] [rbp-360h]
  __m128i v178; // [rsp+630h] [rbp-350h]
  unsigned __int64 *v179; // [rsp+640h] [rbp-340h] BYREF
  __int64 v180; // [rsp+648h] [rbp-338h]
  _BYTE v181[324]; // [rsp+650h] [rbp-330h] BYREF
  int v182; // [rsp+794h] [rbp-1ECh]
  __int64 v183; // [rsp+798h] [rbp-1E8h]
  __int64 v184[2]; // [rsp+7A0h] [rbp-1E0h] BYREF
  _QWORD v185[8]; // [rsp+7B0h] [rbp-1D0h] BYREF
  unsigned __int64 *v186; // [rsp+7F0h] [rbp-190h]
  unsigned int v187; // [rsp+7F8h] [rbp-188h]
  char v188; // [rsp+800h] [rbp-180h] BYREF

  v97 = 0;
  if ( !*(_DWORD *)(*(_QWORD *)(a1 + 128) + 8LL) )
    return v97;
  v2 = a1;
  v136 = 0;
  v128[0] = (unsigned __int64)&v129;
  v128[1] = 0x100000000LL;
  v130 = v132;
  v131 = 0x600000000LL;
  v3 = *(_QWORD *)(a1 + 8);
  v133 = 0;
  v135 = 0;
  v134 = v3;
  HIDWORD(v136) = *(_DWORD *)(v3 + 92);
  sub_B1F440((__int64)v128);
  sub_D51D90((__int64)v137, (__int64)v128);
  v4 = *(_QWORD *)(a1 + 8);
  memset(v166, 0, 96);
  v5 = &v167;
  v166[12] = 1;
  do
  {
    *(_QWORD *)v5 = -4096;
    v5 += 16;
  }
  while ( v5 != (char *)v168 );
  v6 = &v169;
  v168[0] = 0;
  v168[1] = 1;
  do
  {
    *(_QWORD *)v6 = -4096;
    v6 += 24;
    *((_DWORD *)v6 - 4) = 0x7FFFFFFF;
  }
  while ( v6 != &v170 );
  sub_FF9360(v166, v4, (__int64)v137, 0, 0, 0);
  v7 = (__int64 *)sub_22077B0(8u);
  v8 = v7;
  if ( v7 )
    sub_FE7FB0(v7, *(const char **)(a1 + 8), (__int64)v166, (__int64)v137);
  v9 = *(__int64 **)(a1 + 136);
  *(_QWORD *)(a1 + 136) = v8;
  if ( v9 )
  {
    sub_FDC110(v9);
    j_j___libc_free_0((unsigned __int64)v9);
  }
  sub_29B4290(v150, *(_QWORD *)(a1 + 8));
  v101 = 0;
  v104 = &v106;
  v110 = &v112;
  v116 = v118;
  v12 = *(_QWORD *)(a1 + 128);
  v102 = 0;
  v103 = 0;
  v105 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v111 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v117 = 0;
  v100 = 0;
  v106 = 0;
  v112 = 0;
  v13 = *(_QWORD *)v12;
  v14 = *(unsigned int *)(v12 + 8);
  v96 = v13;
  v15 = 3 * v14;
  v95 = v13 + 104 * v14;
  if ( v13 == v95 )
  {
    v97 = *(_DWORD *)(a1 + 24) != 0;
  }
  else
  {
    do
    {
      v122 = (unsigned __int8 **)v124;
      v123 = 0x800000000LL;
      v28 = *(_DWORD *)(v96 + 8);
      if ( v28 )
      {
        sub_26AB6C0((__int64)&v122, v96, v15, v13, v10, v11);
        v98 = &v122[(unsigned int)v123];
        v125 = *(_QWORD *)(v96 + 80);
        v126 = *(_QWORD *)(v96 + 88);
        v127 = *(_QWORD *)(v96 + 96);
        if ( v98 == v122 )
        {
          v28 = 0;
          v16 = 0;
        }
        else
        {
          v29 = (unsigned __int64)v122;
          v28 = 0;
          v16 = 0;
          v30 = v2;
          do
          {
            v31 = *(_QWORD *)v29;
            v32 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(v30 + 168))(
                    *(_QWORD *)(v30 + 176),
                    *(_QWORD *)(*(_QWORD *)v29 + 72LL));
            v33 = sub_26ABAF0(v31, v32);
            if ( v34 == 1 )
              v28 = 1;
            v35 = __OFADD__(v33, v16);
            v16 += v33;
            if ( v35 )
            {
              v16 = 0x8000000000000000LL;
              if ( v33 > 0 )
                v16 = 0x7FFFFFFFFFFFFFFFLL;
            }
            v29 += 8LL;
          }
          while ( v98 != (unsigned __int8 **)v29 );
          v2 = v30;
        }
      }
      else
      {
        v16 = 0;
        v125 = *(_QWORD *)(v96 + 80);
        v126 = *(_QWORD *)(v96 + 88);
        v127 = *(_QWORD *)(v96 + 96);
      }
      v184[0] = (__int64)v185;
      sub_26AB8F0(v184, byte_3F871B3, (__int64)byte_3F871B3);
      v17 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(v2 + 152))(*(_QWORD *)(v2 + 160), *(_QWORD *)(v125 + 72));
      sub_29AFB10(
        (unsigned int)v156,
        (_DWORD)v122,
        v123,
        (unsigned int)v128,
        0,
        *(_QWORD *)(v2 + 136),
        (__int64)v166,
        v17,
        0,
        0,
        0,
        (__int64)v184,
        0);
      if ( (_QWORD *)v184[0] != v185 )
        j_j___libc_free_0(v184[0]);
      sub_29B2CD0(v156, &v100, &v106, &v112, 0);
      if ( !(_DWORD)v111 || (_BYTE)qword_4FF5D88 )
      {
        v18 = sub_29B77F0(v156, v150);
        v20 = v18;
        if ( v18 )
        {
          v21 = *(char **)(*(_QWORD *)(v18 + 16) + 24LL);
          v22 = *v21;
          if ( (unsigned __int8)*v21 <= 0x1Cu || v22 != 85 && v22 != 34 )
            BUG();
          v23 = *(unsigned int *)(v2 + 24);
          v24 = *((_QWORD *)v21 + 5);
          if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 28) )
          {
            v99 = *((_QWORD *)v21 + 5);
            sub_C8D5F0(v2 + 16, (const void *)(v2 + 32), v23 + 1, 0x10u, v24, v19);
            v23 = *(unsigned int *)(v2 + 24);
            v24 = v99;
          }
          v25 = (__int64 *)(*(_QWORD *)(v2 + 16) + 16 * v23);
          *v25 = v20;
          v25[1] = v24;
          ++*(_DWORD *)(v2 + 24);
          if ( v28 == 1 )
            *(_DWORD *)(v2 + 112) = 1;
          v26 = *(_QWORD *)(v2 + 104) + v16;
          if ( __OFADD__(*(_QWORD *)(v2 + 104), v16) )
          {
            v26 = 0x8000000000000000LL;
            if ( v16 > 0 )
              v26 = 0x7FFFFFFFFFFFFFFFLL;
          }
          *(_QWORD *)(v2 + 104) = v26;
          if ( (_BYTE)qword_4FF5CA8 )
          {
            v27 = *(_WORD *)(v20 + 2) & 0xC00F;
            LOBYTE(v27) = v27 | 0x90;
            *(_WORD *)(v20 + 2) = v27;
            *((_WORD *)v21 + 1) = *((_WORD *)v21 + 1) & 0xF003 | 0x24;
          }
        }
        else
        {
          v36 = *(__int64 **)(v2 + 144);
          v37 = *v36;
          v38 = sub_B2BE50(*v36);
          if ( sub_B6EA50(v38)
            || (v93 = sub_B2BE50(v37),
                v94 = sub_B6F970(v93),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v94 + 48LL))(v94)) )
          {
            v39 = *((_QWORD *)*v122 + 7);
            v40 = v39 - 24;
            if ( !v39 )
              v40 = 0;
            sub_B176B0((__int64)v184, (__int64)"partial-inlining", (__int64)"ExtractFailed", 13, v40);
            sub_B18290((__int64)v184, "Failed to extract region at block ", 0x22u);
            sub_B16080((__int64)v118, "Block", 5, *v122);
            v42 = sub_2445430((__int64)v184, (__int64)v118);
            v45 = _mm_loadu_si128((const __m128i *)(v42 + 24));
            v46 = _mm_loadu_si128((const __m128i *)(v42 + 48));
            v172 = *(_DWORD *)(v42 + 8);
            v47 = _mm_loadu_si128((const __m128i *)(v42 + 64));
            v173 = *(_BYTE *)(v42 + 12);
            v48 = *(_QWORD *)(v42 + 16);
            v175 = v45;
            v174 = v48;
            v171 = &unk_49D9D40;
            v49 = *(_QWORD *)(v42 + 40);
            v177 = v46;
            v176 = v49;
            v179 = (unsigned __int64 *)v181;
            v180 = 0x400000000LL;
            v50 = *(unsigned int *)(v42 + 88);
            v178 = v47;
            if ( (_DWORD)v50 )
              sub_26ACA40((__int64)&v179, v42 + 80, v41, v50, v43, v44);
            v181[320] = *(_BYTE *)(v42 + 416);
            v182 = *(_DWORD *)(v42 + 420);
            v183 = *(_QWORD *)(v42 + 424);
            v171 = &unk_49D9DB0;
            if ( v120 != &v121 )
              j_j___libc_free_0((unsigned __int64)v120);
            if ( (__int64 *)v118[0] != &v119 )
              j_j___libc_free_0(v118[0]);
            v51 = v186;
            v184[0] = (__int64)&unk_49D9D40;
            v52 = 10LL * v187;
            v53 = &v186[v52];
            if ( v186 != &v186[v52] )
            {
              do
              {
                v53 -= 10;
                v54 = v53[4];
                if ( (unsigned __int64 *)v54 != v53 + 6 )
                  j_j___libc_free_0(v54);
                if ( (unsigned __int64 *)*v53 != v53 + 2 )
                  j_j___libc_free_0(*v53);
              }
              while ( v51 != v53 );
              v53 = v186;
            }
            if ( v53 != (unsigned __int64 *)&v188 )
              _libc_free((unsigned __int64)v53);
            sub_1049740(v36, (__int64)&v171);
            v55 = v179;
            v171 = &unk_49D9D40;
            v56 = &v179[10 * (unsigned int)v180];
            if ( v179 != v56 )
            {
              do
              {
                v56 -= 10;
                v57 = v56[4];
                if ( (unsigned __int64 *)v57 != v56 + 6 )
                  j_j___libc_free_0(v57);
                if ( (unsigned __int64 *)*v56 != v56 + 2 )
                  j_j___libc_free_0(*v56);
              }
              while ( v55 != v56 );
              v56 = v179;
            }
            if ( v56 != (unsigned __int64 *)v181 )
              _libc_free((unsigned __int64)v56);
          }
        }
      }
      if ( v165 != v166 )
        _libc_free((unsigned __int64)v165);
      sub_C7D6A0(v163[4], 8LL * v164, 8);
      if ( v162 != v163 )
        j_j___libc_free_0((unsigned __int64)v162);
      if ( v160 != &v161 )
        _libc_free((unsigned __int64)v160);
      if ( v159 != (unsigned __int64 *)&v160 )
        _libc_free((unsigned __int64)v159);
      sub_C7D6A0(v157, 8LL * v158, 8);
      if ( v122 != (unsigned __int8 **)v124 )
        _libc_free((unsigned __int64)v122);
      v96 += 104;
    }
    while ( v95 != v96 );
    v97 = *(_DWORD *)(v2 + 24) != 0;
    if ( v116 != v118 )
      _libc_free((unsigned __int64)v116);
  }
  sub_C7D6A0(v113, 8LL * (unsigned int)v115, 8);
  if ( v110 != &v112 )
    _libc_free((unsigned __int64)v110);
  sub_C7D6A0(v107, 8LL * (unsigned int)v109, 8);
  if ( v104 != &v106 )
    _libc_free((unsigned __int64)v104);
  sub_C7D6A0(v101, 8LL * (unsigned int)v103, 8);
  sub_C7D6A0(v154, 8LL * v155, 8);
  v58 = v153;
  if ( v153 )
  {
    v59 = v152;
    v60 = v152 + 40LL * v153;
    do
    {
      if ( *(_QWORD *)v59 != -8192 && *(_QWORD *)v59 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v59 + 16), 8LL * *(unsigned int *)(v59 + 32), 8);
      v59 += 40;
    }
    while ( v60 != v59 );
    v58 = v153;
  }
  v61 = 40 * v58;
  sub_C7D6A0(v152, 40 * v58, 8);
  if ( (char *)v150[0] != &v151 )
    _libc_free(v150[0]);
  sub_D77880((__int64)v166);
  sub_D786F0((__int64)v137);
  v62 = v141;
  v63 = v140;
  if ( v140 != v141 )
  {
    do
    {
      v64 = *(_QWORD *)v63;
      v65 = *(__int64 **)(*(_QWORD *)v63 + 8LL);
      v66 = *(__int64 **)(*(_QWORD *)v63 + 16LL);
      if ( v65 == v66 )
      {
        *(_BYTE *)(v64 + 152) = 1;
      }
      else
      {
        do
        {
          v67 = *v65++;
          sub_D47BB0(v67, v61);
        }
        while ( v66 != v65 );
        *(_BYTE *)(v64 + 152) = 1;
        v68 = *(_QWORD *)(v64 + 8);
        if ( v68 != *(_QWORD *)(v64 + 16) )
          *(_QWORD *)(v64 + 16) = v68;
      }
      v69 = *(_QWORD *)(v64 + 32);
      if ( v69 != *(_QWORD *)(v64 + 40) )
        *(_QWORD *)(v64 + 40) = v69;
      ++*(_QWORD *)(v64 + 56);
      if ( *(_BYTE *)(v64 + 84) )
      {
        *(_QWORD *)v64 = 0;
      }
      else
      {
        v70 = 4 * (*(_DWORD *)(v64 + 76) - *(_DWORD *)(v64 + 80));
        v71 = *(unsigned int *)(v64 + 72);
        if ( v70 < 0x20 )
          v70 = 32;
        if ( (unsigned int)v71 > v70 )
        {
          sub_C8C990(v64 + 56, v61);
        }
        else
        {
          v61 = 0xFFFFFFFFLL;
          memset(*(void **)(v64 + 64), -1, 8 * v71);
        }
        v72 = *(_BYTE *)(v64 + 84);
        *(_QWORD *)v64 = 0;
        if ( !v72 )
          _libc_free(*(_QWORD *)(v64 + 64));
      }
      v73 = *(_QWORD *)(v64 + 32);
      if ( v73 )
      {
        v61 = *(_QWORD *)(v64 + 48) - v73;
        j_j___libc_free_0(v73);
      }
      v74 = *(_QWORD *)(v64 + 8);
      if ( v74 )
      {
        v61 = *(_QWORD *)(v64 + 24) - v74;
        j_j___libc_free_0(v74);
      }
      v63 += 8LL;
    }
    while ( v62 != v63 );
    if ( v140 != v141 )
      v141 = v140;
  }
  v75 = v147;
  v76 = &v147[2 * v148];
  if ( v147 != v76 )
  {
    do
    {
      v77 = v75[1];
      v78 = *v75;
      v75 += 2;
      sub_C7D6A0(v78, v77, 16);
    }
    while ( v76 != v75 );
  }
  v148 = 0;
  if ( v145 )
  {
    v79 = v144;
    v149 = 0;
    v80 = &v144[v145];
    v81 = v144 + 1;
    v142 = *v144;
    for ( i = v142 + 4096; v80 != v81; v79 = v144 )
    {
      v82 = *v81;
      v83 = (unsigned int)(v81 - v79) >> 7;
      v84 = 4096LL << v83;
      if ( v83 >= 0x1E )
        v84 = 0x40000000000LL;
      ++v81;
      sub_C7D6A0(v82, v84, 16);
    }
    v145 = 1;
    sub_C7D6A0(*v79, 4096, 16);
    v85 = v147;
    v86 = &v147[2 * v148];
    if ( v147 == v86 )
      goto LABEL_121;
    do
    {
      v87 = v85[1];
      v88 = *v85;
      v85 += 2;
      sub_C7D6A0(v88, v87, 16);
    }
    while ( v86 != v85 );
  }
  v86 = v147;
LABEL_121:
  if ( v86 != &v149 )
    _libc_free((unsigned __int64)v86);
  if ( v144 != (__int64 *)&v146 )
    _libc_free((unsigned __int64)v144);
  if ( v140 )
    j_j___libc_free_0(v140);
  sub_C7D6A0(v138, 16LL * v139, 8);
  v89 = v130;
  v90 = &v130[8 * (unsigned int)v131];
  if ( v130 != v90 )
  {
    do
    {
      v91 = *((_QWORD *)v90 - 1);
      v90 -= 8;
      if ( v91 )
      {
        v92 = *(_QWORD *)(v91 + 24);
        if ( v92 != v91 + 40 )
          _libc_free(v92);
        j_j___libc_free_0(v91);
      }
    }
    while ( v89 != v90 );
    v90 = v130;
  }
  if ( v90 != v132 )
    _libc_free((unsigned __int64)v90);
  if ( (char *)v128[0] != &v129 )
    _libc_free(v128[0]);
  return v97;
}
