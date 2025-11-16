// Function: sub_2429A80
// Address: 0x2429a80
//
void __fastcall sub_2429A80(unsigned int *a1, unsigned int a2)
{
  unsigned __int32 v3; // ebx
  bool v4; // zf
  __int64 v5; // rdi
  void *v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdi
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r12
  unsigned __int8 *v17; // r13
  size_t v18; // rdx
  size_t v19; // rbx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rdi
  unsigned int v23; // eax
  size_t v24; // rbx
  __int64 v25; // r12
  unsigned __int8 *v26; // r13
  __int64 v27; // rdi
  unsigned int v28; // eax
  unsigned int v29; // eax
  __int64 v30; // rdi
  __int64 v31; // rdi
  unsigned int v32; // eax
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rdi
  void *v36; // rax
  __int64 v37; // rdi
  unsigned int v38; // eax
  unsigned int v39; // eax
  __int64 v40; // rdi
  __int64 v41; // r9
  unsigned int v42; // ebx
  __int64 v43; // rbx
  unsigned __int64 v44; // r14
  __int64 v45; // rdi
  unsigned int v46; // eax
  __int64 v47; // rdi
  unsigned int v48; // eax
  unsigned int v49; // eax
  __int64 v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // r14
  _DWORD *v53; // r14
  unsigned int v54; // eax
  __int64 v55; // rdi
  unsigned int v56; // eax
  __int64 v57; // rdi
  unsigned __int8 *v58; // rbx
  unsigned int v59; // r13d
  int v60; // ecx
  _QWORD *v61; // rsi
  __int64 *v62; // rax
  __int64 v63; // rdx
  __int64 *v64; // r12
  __int64 *v65; // r15
  __int64 v66; // r8
  __int64 v67; // r14
  _BYTE *v68; // rdx
  __int64 v69; // rax
  unsigned int v70; // ecx
  __int64 v71; // rax
  __int64 *v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r13
  __int64 v75; // r15
  __int64 v76; // rdi
  size_t v77; // r12
  __int64 v78; // r14
  unsigned __int8 *v79; // r9
  __int64 v80; // rdi
  unsigned int v81; // eax
  unsigned int *v82; // r12
  unsigned int *v83; // r14
  __int64 v84; // rdx
  unsigned int v85; // eax
  __int64 v86; // rdi
  __int64 v87; // rdi
  __int64 v88; // rdi
  __int64 v89; // rax
  __int64 v90; // rdi
  int v91; // eax
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 v94; // rdx
  unsigned int v95; // eax
  __int64 v96; // rdi
  size_t **v97; // r13
  __int64 v98; // r12
  size_t **v99; // r14
  unsigned __int64 v100; // rax
  size_t **v101; // rbx
  size_t *v102; // r15
  size_t **v103; // r12
  size_t *v104; // rbx
  size_t v105; // r13
  size_t v106; // r14
  size_t v107; // rdx
  int v108; // eax
  __int64 v109; // rdi
  void *v110; // rax
  unsigned int v111; // eax
  __int64 v112; // rdi
  unsigned int v113; // eax
  __int64 v114; // rdi
  _DWORD *v115; // rbx
  _DWORD *v116; // r12
  unsigned int v117; // eax
  __int64 v118; // rdi
  unsigned int v119; // eax
  __int64 v120; // rdi
  __int64 v121; // [rsp+18h] [rbp-248h]
  unsigned __int8 *v122; // [rsp+20h] [rbp-240h]
  __int64 v123; // [rsp+38h] [rbp-228h]
  size_t **v124; // [rsp+48h] [rbp-218h]
  __int64 v125; // [rsp+50h] [rbp-210h]
  _BYTE *v126; // [rsp+50h] [rbp-210h]
  size_t **v127; // [rsp+50h] [rbp-210h]
  _DWORD *s1; // [rsp+58h] [rbp-208h]
  unsigned __int8 *s1b; // [rsp+58h] [rbp-208h]
  void *s1a; // [rsp+58h] [rbp-208h]
  unsigned int v131; // [rsp+6Ch] [rbp-1F4h] BYREF
  unsigned int v132; // [rsp+70h] [rbp-1F0h] BYREF
  char v133; // [rsp+74h] [rbp-1ECh] BYREF
  int v134; // [rsp+78h] [rbp-1E8h] BYREF
  unsigned int v135; // [rsp+7Ch] [rbp-1E4h] BYREF
  unsigned __int8 *v136; // [rsp+80h] [rbp-1E0h] BYREF
  size_t v137; // [rsp+88h] [rbp-1D8h]
  char v138; // [rsp+98h] [rbp-1C8h] BYREF
  _BYTE *v139; // [rsp+120h] [rbp-140h] BYREF
  __int64 v140; // [rsp+128h] [rbp-138h]
  _BYTE v141[304]; // [rsp+130h] [rbp-130h] BYREF

  v3 = a2;
  v4 = *(_DWORD *)(*(_QWORD *)a1 + 72LL) == 1;
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v6 = &loc_1000000;
  if ( !v4 )
    LODWORD(v6) = 1;
  LODWORD(v139) = (_DWORD)v6;
  sub_CB6200(v5, (unsigned __int8 *)&v139, 4u);
  sub_2427090(&v136, *((_BYTE **)a1 + 1));
  sub_2426330(*((_QWORD *)a1 + 1));
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v9 = (v137 >> 2) + (v7 >> 2) + 12;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v9 = _byteswap_ulong(v9);
  LODWORD(v139) = v9;
  sub_CB6200(v8, (unsigned __int8 *)&v139, 4u);
  v10 = a1[5];
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v10 = _byteswap_ulong(v10);
  LODWORD(v139) = v10;
  sub_CB6200(v11, (unsigned __int8 *)&v139, 4u);
  v12 = a1[6];
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v12 = _byteswap_ulong(v12);
  LODWORD(v139) = v12;
  sub_CB6200(v13, (unsigned __int8 *)&v139, 4u);
  v14 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v3 = _byteswap_ulong(a2);
  LODWORD(v139) = v3;
  sub_CB6200(v14, (unsigned __int8 *)&v139, 4u);
  v15 = sub_2426330(*((_QWORD *)a1 + 1));
  v16 = *(_QWORD *)a1;
  v17 = (unsigned __int8 *)v15;
  v19 = v18;
  v20 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v21 = (v18 >> 2) + 1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v21 = _byteswap_ulong(v21);
  LODWORD(v139) = v21;
  sub_CB6200(v20, (unsigned __int8 *)&v139, 4u);
  sub_CB6200(*(_QWORD *)(v16 + 80), v17, v19);
  sub_CB6C70(*(_QWORD *)(v16 + 80), 4 - (v19 & 3));
  v22 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v23 = (*(_DWORD *)(*((_QWORD *)a1 + 1) + 32LL) >> 6) & 1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v23 = _byteswap_ulong(v23);
  LODWORD(v139) = v23;
  sub_CB6200(v22, (unsigned __int8 *)&v139, 4u);
  v24 = v137;
  v25 = *(_QWORD *)a1;
  v26 = v136;
  v27 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v28 = (v137 >> 2) + 1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v28 = _byteswap_ulong(v28);
  LODWORD(v139) = v28;
  sub_CB6200(v27, (unsigned __int8 *)&v139, 4u);
  sub_CB6200(*(_QWORD *)(v25 + 80), v26, v24);
  sub_CB6C70(*(_QWORD *)(v25 + 80), 4 - (v24 & 3));
  v29 = *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL);
  v30 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v29 = _byteswap_ulong(v29);
  LODWORD(v139) = v29;
  sub_CB6200(v30, (unsigned __int8 *)&v139, 4u);
  v31 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  LODWORD(v139) = 0;
  sub_CB6200(v31, (unsigned __int8 *)&v139, 4u);
  v32 = a1[4];
  v33 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v32 = _byteswap_ulong(v32);
  LODWORD(v139) = v32;
  sub_CB6200(v33, (unsigned __int8 *)&v139, 4u);
  v34 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  LODWORD(v139) = 0;
  sub_CB6200(v34, (unsigned __int8 *)&v139, 4u);
  v35 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v36 = &loc_1410000;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    LODWORD(v36) = 16641;
  LODWORD(v139) = (_DWORD)v36;
  sub_CB6200(v35, (unsigned __int8 *)&v139, 4u);
  v37 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v38 = 1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v38 = (unsigned int)&loc_1000000;
  LODWORD(v139) = v38;
  sub_CB6200(v37, (unsigned __int8 *)&v139, 4u);
  v39 = a1[18] + 2;
  v40 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
    v39 = _byteswap_ulong(v39);
  LODWORD(v139) = v39;
  sub_CB6200(v40, (unsigned __int8 *)&v139, 4u);
  v42 = a1[26];
  if ( v42 )
  {
    v109 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
    v110 = &loc_1430000;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
      LODWORD(v110) = 17153;
    LODWORD(v139) = (_DWORD)v110;
    sub_CB6200(v109, (unsigned __int8 *)&v139, 4u);
    v111 = 2 * v42 + 1;
    v112 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
      v111 = _byteswap_ulong(v111);
    LODWORD(v139) = v111;
    sub_CB6200(v112, (unsigned __int8 *)&v139, 4u);
    v113 = a1[22];
    v114 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
      v113 = _byteswap_ulong(v113);
    LODWORD(v139) = v113;
    sub_CB6200(v114, (unsigned __int8 *)&v139, 4u);
    v115 = (_DWORD *)*((_QWORD *)a1 + 12);
    v116 = &v115[4 * a1[26]];
    while ( v116 != v115 )
    {
      v117 = *(_DWORD *)(*(_QWORD *)v115 + 8LL);
      v118 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
      if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
        v117 = _byteswap_ulong(v117);
      LODWORD(v139) = v117;
      sub_CB6200(v118, (unsigned __int8 *)&v139, 4u);
      v119 = v115[2];
      v120 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
      if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
        v119 = _byteswap_ulong(v119);
      v135 = v119;
      v115 += 4;
      sub_CB6200(v120, (unsigned __int8 *)&v135, 4u);
    }
  }
  v43 = *((_QWORD *)a1 + 8);
  v44 = (unsigned __int64)a1[18] << 7;
  v125 = v43 + v44;
  if ( v43 + v44 != v43 )
  {
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)(v43 + 32) )
        {
          v45 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
          v46 = 17153;
          if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) == 1 )
            v46 = (unsigned int)&loc_1430000;
          LODWORD(v139) = v46;
          sub_CB6200(v45, (unsigned __int8 *)&v139, 4u);
          v47 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
          v48 = 2 * *(_DWORD *)(v43 + 32) + 1;
          if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
            v48 = _byteswap_ulong(v48);
          LODWORD(v139) = v48;
          sub_CB6200(v47, (unsigned __int8 *)&v139, 4u);
          v49 = *(_DWORD *)(v43 + 16);
          v50 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
          if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
            v49 = _byteswap_ulong(v49);
          LODWORD(v139) = v49;
          sub_CB6200(v50, (unsigned __int8 *)&v139, 4u);
          v51 = *(_QWORD *)(v43 + 24);
          v52 = 16LL * *(unsigned int *)(v43 + 32);
          s1 = (_DWORD *)(v51 + v52);
          if ( v51 + v52 != v51 )
            break;
        }
        v43 += 128;
        if ( v43 == v125 )
          goto LABEL_46;
      }
      v53 = *(_DWORD **)(v43 + 24);
      do
      {
        v54 = *(_DWORD *)(*(_QWORD *)v53 + 8LL);
        v55 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
        if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
          v54 = _byteswap_ulong(v54);
        LODWORD(v139) = v54;
        sub_CB6200(v55, (unsigned __int8 *)&v139, 4u);
        v56 = v53[2];
        v57 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
        if ( *(_DWORD *)(*(_QWORD *)a1 + 72LL) != 1 )
          v56 = _byteswap_ulong(v56);
        v135 = v56;
        v53 += 4;
        sub_CB6200(v57, (unsigned __int8 *)&v135, 4u);
      }
      while ( s1 != v53 );
      v43 += 128;
    }
    while ( v43 != v125 );
LABEL_46:
    v123 = *((_QWORD *)a1 + 8);
    v121 = v123 + ((unsigned __int64)a1[18] << 7);
    if ( v121 != v123 )
    {
      v58 = (unsigned __int8 *)&v133;
      do
      {
        v59 = 3;
        v139 = v141;
        v140 = 0x2000000000LL;
        v60 = *(_DWORD *)(v123 + 112);
        if ( v60 )
        {
          v61 = *(_QWORD **)(v123 + 104);
          if ( *v61 && *v61 != -8 )
          {
            v64 = *(__int64 **)(v123 + 104);
          }
          else
          {
            v62 = v61 + 1;
            do
            {
              do
              {
                v63 = *v62;
                v64 = v62++;
              }
              while ( !v63 );
            }
            while ( v63 == -8 );
          }
          v65 = &v61[v60];
          if ( v65 == v64 )
          {
            v59 = 3;
          }
          else
          {
            v66 = *v64;
            v67 = *v64;
            v59 = *(_DWORD *)(*v64 + 56) + (*(_QWORD *)(*v64 + 24) >> 2) + 6;
            v68 = v141;
            v69 = 0;
            while ( 1 )
            {
              *(_QWORD *)&v68[8 * v69] = v67;
              v70 = v140 + 1;
              LODWORD(v140) = v140 + 1;
              v71 = v64[1];
              if ( v71 != -8 && v71 )
              {
                ++v64;
              }
              else
              {
                v72 = v64 + 2;
                do
                {
                  do
                  {
                    v73 = *v72;
                    v64 = v72++;
                  }
                  while ( !v73 );
                }
                while ( v73 == -8 );
              }
              if ( v65 == v64 )
                break;
              v67 = *v64;
              v59 += *(_DWORD *)(*v64 + 56) + (*(_QWORD *)(*v64 + 24) >> 2) + 3;
              v69 = v70;
              if ( (unsigned __int64)v70 + 1 > HIDWORD(v140) )
              {
                sub_C8D5F0((__int64)&v139, v141, v70 + 1LL, 8u, v66, v41);
                v69 = (unsigned int)v140;
              }
              v68 = v139;
            }
          }
        }
        v89 = *(_QWORD *)(v123 + 8);
        v4 = *(_DWORD *)(v89 + 72) == 1;
        v90 = *(_QWORD *)(v89 + 80);
        v91 = 21299200;
        if ( !v4 )
          v91 = 17665;
        *(_DWORD *)v58 = v91;
        sub_CB6200(v90, v58, 4u);
        v92 = *(_QWORD *)(v123 + 8);
        v93 = *(_QWORD *)(v92 + 80);
        if ( *(_DWORD *)(v92 + 72) != 1 )
          v59 = _byteswap_ulong(v59);
        v132 = v59;
        sub_CB6200(v93, (unsigned __int8 *)&v132, 4u);
        v94 = *(_QWORD *)(v123 + 8);
        v95 = *(_DWORD *)(v123 + 16);
        v96 = *(_QWORD *)(v94 + 80);
        if ( *(_DWORD *)(v94 + 72) != 1 )
          v95 = _byteswap_ulong(v95);
        v131 = v95;
        sub_CB6200(v96, (unsigned __int8 *)&v131, 4u);
        v97 = (size_t **)v139;
        v98 = 8LL * (unsigned int)v140;
        v124 = (size_t **)&v139[v98];
        v99 = (size_t **)&v139[v98];
        if ( v139 != &v139[v98] )
        {
          _BitScanReverse64(&v100, v98 >> 3);
          sub_2428770((__int64)v139, (size_t **)&v139[v98], 2LL * (int)(63 - (v100 ^ 0x3F)));
          if ( (unsigned __int64)v98 <= 0x80 )
          {
            sub_2425D30(v97, v124);
          }
          else
          {
            sub_2425D30(v97, v97 + 16);
            if ( v99 != v97 + 16 )
            {
              v122 = v58;
              v101 = v97 + 16;
              do
              {
                v102 = *v101;
                v127 = v101;
                v103 = v101;
                s1a = *v101 + 24;
                while ( 1 )
                {
                  while ( 1 )
                  {
                    v104 = *(v103 - 1);
                    v105 = *v102;
                    v106 = *v104;
                    v107 = *v102;
                    if ( *v104 <= *v102 )
                      v107 = *v104;
                    if ( !v107 )
                      break;
                    v108 = memcmp(s1a, v104 + 24, v107);
                    if ( !v108 )
                      break;
                    if ( v108 >= 0 )
                      goto LABEL_91;
                    *v103-- = v104;
                  }
                  if ( v106 == v105 || v106 <= v105 )
                    break;
                  *v103-- = v104;
                }
LABEL_91:
                *v103 = v102;
                v101 = v127 + 1;
              }
              while ( v124 != v127 + 1 );
              v58 = v122;
            }
          }
          v74 = (__int64)v139;
          v126 = &v139[8 * (unsigned int)v140];
          if ( v126 != v139 )
          {
            do
            {
              v75 = *(_QWORD *)v74;
              v76 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v74 + 8LL) + 80LL);
              v132 = 0;
              sub_CB6200(v76, (unsigned __int8 *)&v132, 4u);
              v77 = *(_QWORD *)(v75 + 24);
              v78 = *(_QWORD *)(v75 + 8);
              v79 = *(unsigned __int8 **)(v75 + 16);
              v80 = *(_QWORD *)(v78 + 80);
              v81 = (v77 >> 2) + 1;
              if ( *(_DWORD *)(v78 + 72) != 1 )
                v81 = _byteswap_ulong(v81);
              *(_DWORD *)v58 = v81;
              s1b = v79;
              sub_CB6200(v80, v58, 4u);
              sub_CB6200(*(_QWORD *)(v78 + 80), s1b, v77);
              sub_CB6C70(*(_QWORD *)(v78 + 80), 4 - (v77 & 3));
              v82 = *(unsigned int **)(v75 + 48);
              v83 = &v82[*(unsigned int *)(v75 + 56)];
              while ( v83 != v82 )
              {
                v84 = *(_QWORD *)(v75 + 8);
                v85 = *v82;
                v86 = *(_QWORD *)(v84 + 80);
                if ( *(_DWORD *)(v84 + 72) != 1 )
                  v85 = _byteswap_ulong(v85);
                *(_DWORD *)v58 = v85;
                ++v82;
                sub_CB6200(v86, v58, 4u);
              }
              v74 += 8;
            }
            while ( v126 != (_BYTE *)v74 );
          }
        }
        v87 = *(_QWORD *)(*(_QWORD *)(v123 + 8) + 80LL);
        v135 = 0;
        sub_CB6200(v87, (unsigned __int8 *)&v135, 4u);
        v88 = *(_QWORD *)(*(_QWORD *)(v123 + 8) + 80LL);
        v134 = 0;
        sub_CB6200(v88, (unsigned __int8 *)&v134, 4u);
        if ( v139 != v141 )
          _libc_free((unsigned __int64)v139);
        v123 += 128;
      }
      while ( v123 != v121 );
    }
  }
  if ( v136 != (unsigned __int8 *)&v138 )
    _libc_free((unsigned __int64)v136);
}
