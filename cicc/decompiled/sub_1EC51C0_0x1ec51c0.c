// Function: sub_1EC51C0
// Address: 0x1ec51c0
//
__int64 __fastcall sub_1EC51C0(__int64 a1, __int64 a2, unsigned __int64 *a3, unsigned int *a4, __int64 a5, _BYTE *a6)
{
  __int64 v7; // r14
  int v8; // r10d
  int *v9; // r11
  __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int16 *v12; // rdi
  unsigned int v13; // r12d
  unsigned __int16 *v14; // rsi
  __int64 v16; // rdx
  unsigned __int16 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  char v30; // al
  _QWORD *v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r9
  unsigned int v34; // edx
  _QWORD *v35; // rsi
  _QWORD *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rax
  int v40; // edx
  _BYTE *v41; // r13
  _QWORD *v42; // rdi
  __int64 v43; // r14
  unsigned int *v44; // r12
  __int64 v45; // rsi
  __int64 v46; // r8
  unsigned int v47; // ecx
  __int64 v48; // r15
  __int64 *v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rbx
  __int64 v52; // rcx
  __int64 v53; // r8
  char v54; // r15
  __int64 v55; // r15
  __int64 *v56; // rax
  __int64 v57; // rbx
  __int64 v58; // rdx
  unsigned __int16 *v59; // r8
  unsigned __int16 *v60; // r15
  __int64 v61; // r12
  unsigned __int16 *v62; // r14
  int v63; // eax
  int v64; // r15d
  unsigned __int64 v65; // rbx
  int v66; // r12d
  _QWORD *v67; // r13
  _QWORD *v68; // rbx
  unsigned __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // r13
  unsigned int v72; // r12d
  __int64 v73; // rax
  int v74; // esi
  unsigned int v75; // r14d
  _QWORD *v76; // rax
  _QWORD *v77; // r15
  __int64 v78; // rbx
  __int64 v79; // r12
  __int64 v80; // rbx
  __int64 v81; // rdx
  __int64 v82; // rax
  unsigned __int64 v83; // rdx
  unsigned __int64 v84; // rsi
  unsigned int v85; // r13d
  unsigned int v86; // ecx
  __int64 v87; // rdx
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // r13
  __int64 v91; // r12
  _QWORD *v92; // r14
  __int64 v93; // r15
  __int64 *v94; // rax
  __int64 v95; // rdx
  __int64 v96; // r8
  __int64 v97; // rbx
  __int64 v98; // r10
  __int64 v99; // rsi
  unsigned int v100; // edi
  __int64 v101; // rax
  char v102; // bl
  __int64 v103; // rax
  size_t v104; // rdx
  size_t v105; // r8
  void *v106; // r13
  int v107; // eax
  unsigned int v108; // eax
  __int64 v109; // r15
  unsigned __int64 v110; // rdx
  __int64 v111; // rbx
  unsigned __int16 *v112; // r15
  __int64 v113; // rax
  unsigned int v114; // r12d
  __int64 v115; // r13
  unsigned int v116; // ebx
  unsigned int v117; // r10d
  __int64 v118; // rax
  __int64 *v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // r15
  __int64 v123; // rax
  int v124; // r9d
  __int64 v125; // rdx
  __int64 v126; // r8
  __int64 v127; // rcx
  __int64 v128; // rax
  __int64 v129; // r15
  unsigned __int64 v130; // rdx
  __int64 v131; // rbx
  float v132; // xmm0_4
  __int64 v133; // rax
  _QWORD *v134; // rdx
  __int64 v135; // rsi
  __int64 v136; // rax
  _QWORD *v137; // rdx
  __int64 v138; // rsi
  _BYTE *v139; // [rsp+0h] [rbp-140h]
  unsigned int *v140; // [rsp+8h] [rbp-138h]
  unsigned __int16 *v141; // [rsp+10h] [rbp-130h]
  _QWORD *v142; // [rsp+10h] [rbp-130h]
  __int64 v143; // [rsp+10h] [rbp-130h]
  __int64 v144; // [rsp+18h] [rbp-128h]
  unsigned __int64 v145; // [rsp+18h] [rbp-128h]
  unsigned int *v146; // [rsp+20h] [rbp-120h]
  __int64 v147; // [rsp+20h] [rbp-120h]
  float v148; // [rsp+20h] [rbp-120h]
  _BYTE *v149; // [rsp+28h] [rbp-118h]
  __int64 v150; // [rsp+30h] [rbp-110h]
  unsigned int v151; // [rsp+3Ch] [rbp-104h]
  __int64 v152; // [rsp+50h] [rbp-F0h]
  unsigned int v153; // [rsp+58h] [rbp-E8h]
  int v154; // [rsp+5Ch] [rbp-E4h]
  __int64 v156; // [rsp+70h] [rbp-D0h]
  __int64 v157; // [rsp+78h] [rbp-C8h]
  unsigned int v158; // [rsp+78h] [rbp-C8h]
  __int64 v160; // [rsp+88h] [rbp-B8h]
  bool v161; // [rsp+88h] [rbp-B8h]
  unsigned int v162; // [rsp+90h] [rbp-B0h]
  bool v163; // [rsp+90h] [rbp-B0h]
  __int64 v164; // [rsp+90h] [rbp-B0h]
  __int64 v165; // [rsp+98h] [rbp-A8h]
  unsigned int v166; // [rsp+98h] [rbp-A8h]
  __int64 v167; // [rsp+98h] [rbp-A8h]
  size_t v168; // [rsp+98h] [rbp-A8h]
  _QWORD *n; // [rsp+A0h] [rbp-A0h]
  size_t na; // [rsp+A0h] [rbp-A0h]
  size_t nb; // [rsp+A0h] [rbp-A0h]
  size_t nc; // [rsp+A0h] [rbp-A0h]
  size_t nd; // [rsp+A0h] [rbp-A0h]
  size_t ne; // [rsp+A0h] [rbp-A0h]
  char v175; // [rsp+AAh] [rbp-96h]
  char v176; // [rsp+ABh] [rbp-95h]
  unsigned int v177; // [rsp+ACh] [rbp-94h]
  unsigned __int64 v178; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v179; // [rsp+B8h] [rbp-88h] BYREF
  __int64 v180; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v181; // [rsp+C8h] [rbp-78h]
  __int64 v182; // [rsp+D0h] [rbp-70h]
  __int64 v183; // [rsp+D8h] [rbp-68h]
  __int64 v184; // [rsp+E0h] [rbp-60h]
  __int64 v185; // [rsp+E8h] [rbp-58h]
  __int64 v186; // [rsp+F0h] [rbp-50h]
  __int64 v187; // [rsp+F8h] [rbp-48h]
  int v188; // [rsp+100h] [rbp-40h]
  float (__fastcall *v189)(int, float); // [rsp+108h] [rbp-38h]

  v7 = a2;
  v149 = a6;
  LODWORD(a6) = -*(_DWORD *)(a2 + 8);
  v176 = a5;
  *(_DWORD *)(a2 + 64) = (_DWORD)a6;
  v177 = -1;
  while ( 1 )
  {
    if ( (int)a6 < 0 )
    {
      v16 = *(unsigned int *)(v7 + 8);
      v17 = *(unsigned __int16 **)v7;
      *(_DWORD *)(v7 + 64) = (_DWORD)a6 + 1;
      a6 = (_BYTE *)(v16 + (int)a6);
      v162 = v17[(_QWORD)a6];
    }
    else
    {
      if ( *(_BYTE *)(v7 + 68) )
        return v177;
      v8 = *(_DWORD *)(v7 + 56);
      v9 = (int *)&v180;
      v10 = 2LL * (int)a6;
      do
      {
        if ( (int)a6 >= v8 )
          return v177;
        v11 = *(_QWORD *)(v7 + 48);
        v12 = *(unsigned __int16 **)v7;
        *(_DWORD *)(v7 + 64) = (_DWORD)a6 + 1;
        v13 = *(unsigned __int16 *)(v11 + v10);
        v10 += 2;
        v14 = &v12[*(unsigned int *)(v7 + 8)];
        LODWORD(v180) = v13;
      }
      while ( v14 != sub_1EBB4B0(v12, (__int64)v14, v9) );
      v162 = v13;
    }
    if ( !v162 )
      return v177;
    if ( v176
      && v162 < *(_DWORD *)(a1 + 328)
      && *(_WORD *)(*(_QWORD *)(a1 + 320) + 2LL * v162)
      && !(unsigned __int8)sub_2103340(*(_QWORD *)(a1 + 272)) )
    {
      LODWORD(a6) = *(_DWORD *)(v7 + 64);
    }
    else
    {
      v18 = a1 + 1000;
      v160 = a1 + 1000;
      v19 = *a4;
      if ( (_DWORD)v19 != 32 )
        goto LABEL_12;
      v158 = 0;
      v71 = 0;
      v72 = 0;
      v166 = -1;
      na = a1;
      v156 = v7;
      do
      {
        if ( v177 != v72 )
        {
          v18 = na;
          v73 = v71 + *(_QWORD *)(na + 24168);
          if ( *(_DWORD *)v73 )
          {
            v74 = *(_DWORD *)(v73 + 40);
            v18 = (unsigned int)(v74 + 63) >> 6;
            v75 = (unsigned int)(v74 + 63) >> 6;
            if ( v75 )
            {
              v76 = *(_QWORD **)(v73 + 24);
              v75 = 0;
              v77 = v76 + 1;
              v78 = (__int64)&v76[(unsigned int)(v18 - 1) + 1];
              while ( 1 )
              {
                v75 += sub_39FAC40(*v76);
                v76 = v77;
                if ( (_QWORD *)v78 == v77 )
                  break;
                ++v77;
              }
            }
            if ( v75 < v166 )
            {
              v158 = v72;
              v166 = v75;
            }
          }
        }
        ++v72;
        v71 += 96;
      }
      while ( v72 != 32 );
      a1 = na;
      v7 = v156;
      *a4 = 31;
      v79 = *(_QWORD *)(na + 24168);
      v80 = v79 + 96LL * v158;
      *(_DWORD *)v80 = *(_DWORD *)(v79 + 2976);
      v81 = *(_QWORD *)(v80 + 8);
      *(_DWORD *)(v80 + 4) = *(_DWORD *)(v79 + 2980);
      v82 = *(_QWORD *)(v79 + 2984);
      *(_QWORD *)(v80 + 16) = 0;
      if ( v81 )
        --*(_DWORD *)(v81 + 8);
      *(_QWORD *)(v80 + 8) = v82;
      if ( v82 )
        ++*(_DWORD *)(v82 + 8);
      v83 = v79 + 3000;
      if ( v79 + 3000 != v80 + 24 )
      {
        v84 = *(unsigned int *)(v79 + 3016);
        v83 = *(_QWORD *)(v80 + 32);
        *(_DWORD *)(v80 + 40) = v84;
        v85 = (unsigned int)(v84 + 63) >> 6;
        v18 = v83 << 6;
        a5 = v85;
        if ( v84 > v83 << 6 )
        {
          nc = 8LL * v85;
          v103 = malloc(nc);
          v104 = nc;
          v105 = (unsigned int)(v84 + 63) >> 6;
          v106 = (void *)v103;
          if ( !v103 )
          {
            if ( nc || (v136 = malloc(1u), v104 = 0, v105 = (unsigned int)(v84 + 63) >> 6, !v136) )
            {
              v168 = v105;
              ne = v104;
              sub_16BD1C0("Allocation failed", 1u);
              v104 = ne;
              v105 = v168;
            }
            else
            {
              v106 = (void *)v136;
            }
          }
          nd = v105;
          memcpy(v106, *(const void **)(v79 + 3000), v104);
          _libc_free(*(_QWORD *)(v80 + 24));
          LOBYTE(a5) = nd;
          *(_QWORD *)(v80 + 24) = v106;
          *(_QWORD *)(v80 + 32) = nd;
          goto LABEL_98;
        }
        if ( (_DWORD)v84 )
        {
          memcpy(*(void **)(v80 + 24), *(const void **)(v79 + 3000), 8LL * v85);
          v107 = *(_DWORD *)(v80 + 40);
          v83 = *(_QWORD *)(v80 + 32);
          v85 = (unsigned int)(v107 + 63) >> 6;
          a5 = v85;
          if ( v83 > v85 )
          {
LABEL_129:
            v83 -= a5;
            if ( v83 )
              memset((void *)(*(_QWORD *)(v80 + 24) + 8 * a5), 0, 8 * v83);
            v107 = *(_DWORD *)(v80 + 40);
          }
          v108 = v107 & 0x3F;
          if ( v108 )
          {
            v18 = v108;
            v83 = -1LL << v108;
            *(_QWORD *)(*(_QWORD *)(v80 + 24) + 8LL * (v85 - 1)) &= ~(-1LL << v108);
          }
          goto LABEL_98;
        }
        if ( v83 > v85 )
          goto LABEL_129;
      }
LABEL_98:
      sub_1EBB700(v80 + 48, v79 + 3024, v83, v18, (unsigned __int8)a5, (int)a6);
      v86 = v158;
      v19 = *a4;
      if ( v177 != (_DWORD)v19 )
        v86 = v177;
      v177 = v86;
LABEL_12:
      v20 = *(unsigned int *)(a1 + 24176);
      if ( (unsigned int)v20 > (unsigned int)v19 )
        goto LABEL_13;
      v65 = (unsigned int)(v19 + 1);
      v66 = v19 + 1;
      if ( v65 >= v20 )
      {
        if ( v65 <= v20 )
        {
LABEL_13:
          v21 = *(_QWORD *)(a1 + 24168);
          goto LABEL_14;
        }
        if ( v65 > *(unsigned int *)(a1 + 24180) )
        {
          sub_1EBCFA0(a1 + 24168, v65);
          v20 = *(unsigned int *)(a1 + 24176);
        }
        v21 = *(_QWORD *)(a1 + 24168);
        v87 = v21 + 96 * v20;
        v88 = v21 + 96 * v65;
        if ( v87 != v88 )
        {
          do
          {
            if ( v87 )
            {
              memset((void *)v87, 0, 0x60u);
              *(_DWORD *)(v87 + 60) = 8;
              *(_QWORD *)(v87 + 48) = v87 + 64;
            }
            v87 += 96;
          }
          while ( v88 != v87 );
          goto LABEL_77;
        }
      }
      else
      {
        v21 = *(_QWORD *)(a1 + 24168);
        v67 = (_QWORD *)(v21 + 96 * v20);
        v68 = (_QWORD *)(v21 + 96 * v65);
        if ( v67 != v68 )
        {
          do
          {
            v67 -= 12;
            v69 = v67[6];
            if ( (_QWORD *)v69 != v67 + 8 )
              _libc_free(v69);
            _libc_free(v67[3]);
            v70 = v67[1];
            v67[2] = 0;
            if ( v70 )
              --*(_DWORD *)(v70 + 8);
          }
          while ( v68 != v67 );
LABEL_77:
          v21 = *(_QWORD *)(a1 + 24168);
        }
      }
      *(_DWORD *)(a1 + 24176) = v66;
      v19 = *a4;
LABEL_14:
      v22 = v21 + 96 * v19;
      *(_DWORD *)(v22 + 4) = 0;
      *(_DWORD *)v22 = v162;
      v23 = *(_QWORD *)(v22 + 8);
      *(_QWORD *)(v22 + 16) = 0;
      if ( v23 )
        --*(_DWORD *)(v23 + 8);
      *(_QWORD *)(v22 + 8) = 0;
      v24 = sub_20F81F0(v160, v162);
      v25 = *(_QWORD *)(v22 + 8);
      *(_QWORD *)(v22 + 16) = 0;
      if ( v25 )
        --*(_DWORD *)(v25 + 8);
      *(_QWORD *)(v22 + 8) = v24;
      if ( v24 )
        ++*(_DWORD *)(v24 + 8);
      *(_DWORD *)(v22 + 40) = 0;
      *(_DWORD *)(v22 + 56) = 0;
      sub_1F130D0(*(_QWORD *)(a1 + 848), v22 + 24);
      v181 = 0;
      v29 = *(_QWORD *)(v22 + 8);
      v178 = 0;
      v180 = v29;
      if ( v29 )
        ++*(_DWORD *)(v29 + 8);
      v30 = sub_1EBCB20(a1, &v180, &v178, v26, v27, v28);
      v32 = v180;
      v181 = 0;
      if ( v180 )
        --*(_DWORD *)(v180 + 8);
      if ( v30 )
      {
        if ( *a3 > v178 )
        {
          sub_1EBC180((_QWORD *)a1, v22, v32, v31);
          sub_1F13350(*(_QWORD *)(a1 + 848));
          v34 = (unsigned int)(*(_DWORD *)(v22 + 40) + 63) >> 6;
          if ( v34 )
          {
            v35 = *(_QWORD **)(v22 + 24);
            v36 = v35;
            v37 = (__int64)&v35[v34];
            while ( !*v36 )
            {
              if ( (_QWORD *)v37 == ++v36 )
                goto LABEL_61;
            }
            v38 = *(_QWORD *)(a1 + 984);
            v179 = 0;
            v154 = *(_DWORD *)(*(_QWORD *)(v38 + 40) + 112LL);
            v39 = *(_QWORD *)(v38 + 280);
            v40 = *(_DWORD *)(v38 + 288);
            if ( v40 )
            {
              v41 = (_BYTE *)(v39 + 33);
              v165 = 0;
              v175 = 0;
              v157 = 8LL * (unsigned int)(v40 - 1);
              n = (_QWORD *)v22;
              v151 = v154 & 0x7FFFFFFF;
              v42 = v35;
              v153 = (v154 & 0x7FFFFFFF) + 1;
              v150 = 8LL * v153;
              v152 = v7;
              v43 = a1;
              while ( 1 )
              {
                v44 = (unsigned int *)(*(_QWORD *)(v43 + 24088) + v165);
                v45 = *v44;
                v46 = *(_QWORD *)(*(_QWORD *)(v43 + 840) + 240LL);
                v33 = (unsigned int)(2 * v45);
                v47 = *(_DWORD *)(v46 + 4LL * (unsigned int)(v33 + 1));
                v163 = (v42[*(_DWORD *)(v46 + 4 * v33) >> 6] & (1LL << *(_DWORD *)(v46 + 4 * v33))) != 0;
                v161 = (v42[v47 >> 6] & (1LL << v47)) != 0;
                v48 = n[1];
                v49 = &qword_4FCF930;
                if ( v48 )
                {
                  v50 = *(unsigned int *)(v48 + 4);
                  v51 = 24LL * (unsigned int)v45;
                  v49 = (__int64 *)(v51 + *(_QWORD *)(v48 + 512));
                  if ( *(_DWORD *)v49 != (_DWORD)v50 )
                  {
                    sub_20F85B0(n[1], v45, 3LL * (unsigned int)v45, v50, v46, v33);
                    v49 = (__int64 *)(v51 + *(_QWORD *)(v48 + 512));
                  }
                }
                n[2] = v49;
                if ( !*(_BYTE *)(v43 + 27409) || (v49[1] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                  goto LABEL_50;
                if ( !*(v41 - 1) )
                  goto LABEL_64;
                if ( *v41 )
                  break;
                v63 = v163 ^ (*((_BYTE *)v44 + 4) == 1);
LABEL_53:
                v64 = v63 - 1;
                if ( v63 )
                {
                  while ( 1 )
                  {
                    sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 848) + 376LL) + 8LL * *v44));
                    if ( !v64 )
                      break;
                    v64 = 0;
                  }
                }
LABEL_62:
                v41 += 40;
                if ( v157 == v165 )
                {
                  a1 = v43;
                  v22 = (__int64)n;
                  v7 = v152;
                  goto LABEL_110;
                }
                v165 += 8;
                v42 = (_QWORD *)n[3];
              }
              if ( !v163 || !v161 )
              {
                v63 = v163 ^ (*((_BYTE *)v44 + 4) == 1);
                goto LABEL_52;
              }
              v54 = sub_1EC4940(v43, v154, (__int64)n, *v44, v152, v33);
              if ( v54 )
              {
                sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 848) + 376LL) + 8LL * *v44));
                sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 848) + 376LL) + 8LL * *v44));
                v175 = v54;
                goto LABEL_50;
              }
              v55 = n[1];
              v56 = &qword_4FCF930;
              if ( v55 )
              {
                v57 = 24LL * *v44;
                v58 = *(unsigned int *)(v55 + 4);
                v56 = (__int64 *)(v57 + *(_QWORD *)(v55 + 512));
                if ( *(_DWORD *)v56 != (_DWORD)v58 )
                {
                  sub_20F85B0(v55, *v44, v58, v52, v53, v33);
                  v56 = (__int64 *)(v57 + *(_QWORD *)(v55 + 512));
                }
              }
              n[2] = v56;
              v59 = *(unsigned __int16 **)(v152 + 48);
              v60 = &v59[*(_QWORD *)(v152 + 56)];
              if ( v59 != v60 )
              {
                v146 = v44;
                v61 = v43;
                v62 = *(unsigned __int16 **)(v152 + 48);
                do
                {
                  if ( !(unsigned __int8)sub_2103AA0(
                                           *(_QWORD *)(v61 + 272),
                                           v56[1] & 6
                                         | *(_QWORD *)(v56[1] & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL,
                                           v56[2],
                                           *v62) )
                  {
                    v43 = v61;
                    v44 = v146;
                    goto LABEL_50;
                  }
                  ++v62;
                  v56 = (__int64 *)n[2];
                }
                while ( v60 != v62 );
                v43 = v61;
                v44 = v146;
              }
              v109 = *(_QWORD *)(v43 + 264);
              v147 = v56[2];
              v110 = *(unsigned int *)(v109 + 408);
              v144 = v56[1];
              if ( v151 < (unsigned int)v110 )
              {
                v111 = *(_QWORD *)(*(_QWORD *)(v109 + 400) + 8LL * (v154 & 0x7FFFFFFF));
                if ( v111 )
                {
LABEL_137:
                  LODWORD(v180) = -1;
                  v112 = *(unsigned __int16 **)(v152 + 48);
                  v113 = *(_QWORD *)(v152 + 56);
                  HIDWORD(v180) = *(_DWORD *)(v111 + 116);
                  v141 = &v112[v113];
                  if ( v112 == v141 )
                    goto LABEL_143;
                  v140 = v44;
                  v139 = v41;
                  v114 = 0;
                  v115 = v111;
                  do
                  {
                    v116 = *v112;
                    if ( (unsigned __int8)sub_1EBC970((_QWORD *)v43, v115, v116, v144, v147, (__int64)&v180) )
                      v114 = v116;
                    ++v112;
                  }
                  while ( v141 != v112 );
                  v117 = v114;
                  v41 = v139;
                  v44 = v140;
                  v148 = *((float *)&v180 + 1);
                  if ( !v117 )
                  {
LABEL_143:
                    sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 848) + 376LL) + 8LL * *v44));
                    sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 848) + 376LL) + 8LL * *v44));
                    goto LABEL_50;
                  }
                  v119 = *(__int64 **)(v43 + 8);
                  v120 = *v119;
                  v121 = v119[1];
                  if ( v120 == v121 )
LABEL_184:
                    BUG();
                  while ( *(_UNKNOWN **)v120 != &unk_4FC6A0C )
                  {
                    v120 += 16;
                    if ( v121 == v120 )
                      goto LABEL_184;
                  }
                  v122 = *(_QWORD *)(v43 + 808);
                  v123 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v120 + 8) + 104LL))(
                           *(_QWORD *)(v120 + 8),
                           &unk_4FC6A0C);
                  v125 = *(_QWORD *)(v43 + 256);
                  v126 = *(_QWORD *)(v43 + 264);
                  v184 = v122;
                  v183 = v123;
                  v127 = *(_QWORD *)(v43 + 680);
                  v189 = sub_1EBAF90;
                  v182 = v125;
                  v180 = v127;
                  v181 = v126;
                  v185 = 0;
                  v186 = 0;
                  v187 = 0;
                  v188 = 0;
                  v128 = n[2];
                  v129 = *(_QWORD *)(v128 + 16);
                  v145 = *(_QWORD *)(v128 + 8) & 6LL
                       | *(_QWORD *)(*(_QWORD *)(v128 + 8) & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL;
                  v130 = *(unsigned int *)(v126 + 408);
                  if ( v151 < (unsigned int)v130 )
                  {
                    v131 = *(_QWORD *)(*(_QWORD *)(v126 + 400) + 8LL * (v154 & 0x7FFFFFFF));
                    if ( v131 )
                    {
LABEL_157:
                      v132 = sub_20E3050(&v180, v131, v145, v129);
                      if ( v132 < 0.0 || v132 <= v148 )
                      {
                        j___libc_free_0(v186);
                        goto LABEL_143;
                      }
                      j___libc_free_0(v186);
LABEL_50:
                      if ( *(v41 - 1) )
                      {
                        v63 = v163 ^ (*((_BYTE *)v44 + 4) == 1);
                        if ( !*v41 )
                          goto LABEL_53;
                      }
                      else
                      {
LABEL_64:
                        if ( !*v41 )
                          goto LABEL_62;
                        v63 = 0;
                      }
LABEL_52:
                      v63 += v161 ^ (*((_BYTE *)v44 + 5) == 1);
                      goto LABEL_53;
                    }
                  }
                  if ( v153 > (unsigned int)v130 )
                  {
                    if ( v153 >= v130 )
                    {
                      if ( v153 <= v130 )
                        goto LABEL_161;
                      if ( v153 > (unsigned __int64)*(unsigned int *)(v126 + 412) )
                      {
                        v143 = v126;
                        sub_16CD150(v126 + 400, (const void *)(v126 + 416), v153, 8, v126, v124);
                        v126 = v143;
                        v130 = *(unsigned int *)(v143 + 408);
                      }
                      v133 = *(_QWORD *)(v126 + 400);
                      v137 = (_QWORD *)(v133 + 8 * v130);
                      v138 = *(_QWORD *)(v126 + 416);
                      if ( (_QWORD *)(v133 + v150) != v137 )
                      {
                        do
                          *v137++ = v138;
                        while ( (_QWORD *)(v133 + v150) != v137 );
                        v133 = *(_QWORD *)(v126 + 400);
                      }
                      *(_DWORD *)(v126 + 408) = v153;
                    }
                    else
                    {
                      *(_DWORD *)(v126 + 408) = v153;
                      v133 = *(_QWORD *)(v126 + 400);
                    }
                  }
                  else
                  {
LABEL_161:
                    v133 = *(_QWORD *)(v126 + 400);
                  }
                  v142 = (_QWORD *)v126;
                  *(_QWORD *)(v133 + 8LL * (v154 & 0x7FFFFFFF)) = sub_1DBA290(v154);
                  v131 = *(_QWORD *)(v142[50] + 8LL * (v154 & 0x7FFFFFFF));
                  sub_1DBB110(v142, v131);
                  goto LABEL_157;
                }
              }
              if ( (unsigned int)v110 < v153 )
              {
                if ( v153 >= v110 )
                {
                  if ( v153 <= v110 )
                    goto LABEL_145;
                  if ( v153 > (unsigned __int64)*(unsigned int *)(v109 + 412) )
                  {
                    sub_16CD150(v109 + 400, (const void *)(v109 + 416), v153, 8, (int)v59, v33);
                    v110 = *(unsigned int *)(v109 + 408);
                  }
                  v118 = *(_QWORD *)(v109 + 400);
                  v134 = (_QWORD *)(v118 + 8 * v110);
                  v135 = *(_QWORD *)(v109 + 416);
                  if ( (_QWORD *)(v118 + v150) != v134 )
                  {
                    do
                      *v134++ = v135;
                    while ( (_QWORD *)(v118 + v150) != v134 );
                    v118 = *(_QWORD *)(v109 + 400);
                  }
                  *(_DWORD *)(v109 + 408) = v153;
                }
                else
                {
                  *(_DWORD *)(v109 + 408) = v153;
                  v118 = *(_QWORD *)(v109 + 400);
                }
              }
              else
              {
LABEL_145:
                v118 = *(_QWORD *)(v109 + 400);
              }
              *(_QWORD *)(v118 + 8LL * (v154 & 0x7FFFFFFF)) = sub_1DBA290(v154);
              v111 = *(_QWORD *)(*(_QWORD *)(v109 + 400) + 8LL * (v154 & 0x7FFFFFFF));
              sub_1DBB110((_QWORD *)v109, v111);
              goto LABEL_137;
            }
            v175 = 0;
LABEL_110:
            v89 = *(unsigned int *)(v22 + 56);
            if ( (_DWORD)v89 )
            {
              v167 = v7;
              v90 = 0;
              v91 = a1;
              nb = 4 * v89;
              v92 = (_QWORD *)v22;
              do
              {
                v96 = v92[3];
                v97 = *(unsigned int *)(v92[6] + v90);
                v98 = *(_QWORD *)(*(_QWORD *)(v91 + 840) + 240LL);
                v99 = *(_QWORD *)(v96 + 8LL * (*(_DWORD *)(v98 + 4LL * (unsigned int)(2 * v97)) >> 6))
                    & (1LL << *(_DWORD *)(v98 + 4LL * (unsigned int)(2 * v97)));
                v100 = *(_DWORD *)(v98 + 4LL * (unsigned int)(2 * v97 + 1));
                v101 = *(_QWORD *)(v96 + 8LL * (v100 >> 6)) & (1LL << v100);
                if ( v101 | v99 )
                {
                  if ( v99 && v101 )
                  {
                    v93 = v92[1];
                    v94 = &qword_4FCF930;
                    if ( v93 )
                    {
                      v95 = *(unsigned int *)(v93 + 4);
                      v94 = (__int64 *)(*(_QWORD *)(v93 + 512) + 24LL * (unsigned int)v97);
                      if ( *(_DWORD *)v94 != (_DWORD)v95 )
                      {
                        sub_20F85B0(v93, (unsigned int)v97, v95, 24LL * (unsigned int)v97, v96, v33);
                        v94 = (__int64 *)(24LL * (unsigned int)v97 + *(_QWORD *)(v93 + 512));
                      }
                    }
                    v92[2] = v94;
                    if ( (v94[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                    {
                      v164 = (unsigned int)v97;
                      sub_16AF570(
                        &v179,
                        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v91 + 848) + 376LL) + 8LL * (unsigned int)v97));
                      sub_16AF570(
                        &v179,
                        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v91 + 848) + 376LL) + 8LL * (unsigned int)v97));
                      if ( *(_BYTE *)(v91 + 27409) )
                      {
                        v102 = sub_1EC4940(v91, v154, (__int64)v92, v97, v167, v33);
                        if ( v102 )
                        {
                          sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v91 + 848) + 376LL) + 8 * v164));
                          sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v91 + 848) + 376LL) + 8 * v164));
                          v175 = v102;
                        }
                      }
                    }
                  }
                  else
                  {
                    sub_16AF570(&v179, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v91 + 848) + 376LL) + 8 * v97));
                  }
                }
                v90 += 4;
              }
              while ( nb != v90 );
              v7 = v167;
              a1 = v91;
            }
            sub_16AF570(&v178, v179);
            if ( *a3 > v178 )
            {
              v177 = *a4;
              *a3 = v178;
              if ( v149 )
                *v149 = v175;
            }
            ++*a4;
          }
        }
      }
LABEL_61:
      LODWORD(a6) = *(_DWORD *)(v7 + 64);
    }
  }
}
