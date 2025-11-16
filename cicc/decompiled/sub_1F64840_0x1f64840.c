// Function: sub_1F64840
// Address: 0x1f64840
//
void __fastcall sub_1F64840(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, int a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 *v12; // r8
  __int64 *v13; // rax
  __int64 *i; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // r14
  _QWORD *v20; // rax
  unsigned int v21; // esi
  __int64 v22; // rdi
  __int64 v23; // r8
  unsigned int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r13
  _QWORD *v28; // rax
  __int64 *v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  _QWORD *v34; // rax
  unsigned __int64 v35; // r13
  unsigned int v36; // r12d
  unsigned int v37; // esi
  __int64 v38; // r15
  __int64 v39; // r8
  unsigned int v40; // edi
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r13
  _QWORD *v44; // r15
  char v45; // al
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 *v53; // rbx
  int v54; // eax
  __int64 v55; // rcx
  __int64 *v56; // r13
  __int64 v57; // rsi
  __int64 v58; // rdx
  __int64 v59; // rcx
  _QWORD *v60; // rax
  __int64 v61; // rax
  int v62; // r8d
  __int64 v63; // rax
  __m128i v64; // xmm1
  __m128i *v65; // rax
  __int64 v66; // r14
  __int64 v67; // rax
  __int64 v68; // r15
  unsigned int v69; // eax
  unsigned __int64 v70; // rbx
  void *v71; // r8
  void **v72; // rdi
  unsigned int v73; // eax
  int v74; // r12d
  _BYTE *v75; // rdi
  size_t v76; // rdx
  int v77; // eax
  int v78; // edx
  __int64 v79; // rsi
  unsigned int v80; // eax
  __int64 v81; // rcx
  int v82; // edi
  __int64 v83; // rax
  _QWORD *v84; // rax
  unsigned int v85; // r12d
  __int64 v86; // r13
  _QWORD *v87; // rax
  __int64 v88; // rdi
  __int64 v89; // rax
  __int64 j; // rbx
  unsigned int v91; // ecx
  int v92; // r13d
  __int64 v93; // r10
  int v94; // ecx
  int v95; // ecx
  int v96; // esi
  int v97; // esi
  __int64 v98; // r8
  __int64 v99; // rdx
  __int64 v100; // rdi
  int v101; // r11d
  int v102; // esi
  int v103; // esi
  __int64 v104; // r8
  int v105; // r11d
  __int64 v106; // rdx
  __int64 v107; // rdi
  int v108; // r11d
  int v109; // ecx
  int v110; // edi
  int v111; // edi
  __int64 v112; // rdx
  __int64 v113; // rsi
  int v114; // r10d
  int v115; // edx
  int v116; // edx
  int v117; // edi
  __int64 v118; // r13
  __int64 v119; // rsi
  __int64 v120; // [rsp+8h] [rbp-E8h]
  unsigned int v121; // [rsp+10h] [rbp-E0h]
  __int64 *v122; // [rsp+20h] [rbp-D0h]
  unsigned int v123; // [rsp+28h] [rbp-C8h]
  __int64 v124; // [rsp+28h] [rbp-C8h]
  __int64 *v125; // [rsp+30h] [rbp-C0h]
  __int64 *v127; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v128; // [rsp+48h] [rbp-A8h]
  _BYTE v129[16]; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v130; // [rsp+60h] [rbp-90h] BYREF
  __m128i v131; // [rsp+70h] [rbp-80h] BYREF
  __int64 v132; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v133; // [rsp+88h] [rbp-68h]
  void *src; // [rsp+90h] [rbp-60h] BYREF
  __int64 v135; // [rsp+98h] [rbp-58h]
  _BYTE v136[80]; // [rsp+A0h] [rbp-50h] BYREF

  v6 = a3;
  v8 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)(a2 + 16) == 34 )
  {
    v127 = (__int64 *)v129;
    v128 = 0x200000000LL;
    v9 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v10 = *(_QWORD *)(a2 - 8);
      v11 = (__int64 *)(v10 + v9);
    }
    else
    {
      v10 = a2 - v9;
      v11 = (__int64 *)a2;
    }
    v12 = (__int64 *)(v10 + 24);
    v13 = (__int64 *)(v10 + 48);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      v12 = v13;
    for ( i = v12; v11 != i; LODWORD(v128) = v128 + 1 )
    {
      v15 = sub_1523720(*i);
      a6 = sub_157ED20(v15);
      v16 = (unsigned int)v128;
      if ( (unsigned int)v128 >= HIDWORD(v128) )
      {
        v124 = a6;
        sub_16CD150((__int64)&v127, v129, 0, 8, (int)v12, a6);
        v16 = (unsigned int)v128;
        a6 = v124;
      }
      i += 3;
      v127[v16] = a6;
    }
    v17 = (unsigned int)v6;
    v18 = *(unsigned int *)(a1 + 136);
    v19 = a1 + 128;
    if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 140) )
    {
      sub_16CD150(a1 + 128, (const void *)(a1 + 144), 0, 16, (int)v12, a6);
      v18 = *(unsigned int *)(a1 + 136);
    }
    v20 = (_QWORD *)(*(_QWORD *)(a1 + 128) + 16 * v18);
    *v20 = v6;
    v20[1] = 0;
    v123 = *(_DWORD *)(a1 + 136);
    *(_DWORD *)(a1 + 136) = v123 + 1;
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = *(_QWORD *)(a1 + 8);
      LODWORD(v23) = v21 - 1;
      v24 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = v22 + 16LL * v24;
      v26 = *(_QWORD *)v25;
      if ( a2 == *(_QWORD *)v25 )
      {
LABEL_14:
        *(_DWORD *)(v25 + 8) = v123;
        v27 = *(_QWORD *)(v8 + 8);
        if ( v27 )
        {
          while ( 1 )
          {
            v28 = sub_1648700(v27);
            if ( (unsigned __int8)(*((_BYTE *)v28 + 16) - 25) <= 9u )
              break;
            v27 = *(_QWORD *)(v27 + 8);
            if ( !v27 )
              goto LABEL_26;
          }
LABEL_22:
          v32 = v28[5];
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            v29 = *(__int64 **)(a2 - 8);
          else
            v29 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
          v30 = sub_1F60340(v32, *v29);
          if ( v30 )
          {
            v31 = sub_157ED20(v30);
            sub_1F64840(a1, v31, v123);
          }
          while ( 1 )
          {
            v27 = *(_QWORD *)(v27 + 8);
            if ( !v27 )
              break;
            v28 = sub_1648700(v27);
            if ( (unsigned __int8)(*((_BYTE *)v28 + 16) - 25) <= 9u )
              goto LABEL_22;
          }
        }
LABEL_26:
        v33 = *(unsigned int *)(a1 + 136);
        if ( (unsigned int)v33 >= *(_DWORD *)(a1 + 140) )
        {
          sub_16CD150(v19, (const void *)(a1 + 144), 0, 16, v23, a6);
          v33 = *(unsigned int *)(a1 + 136);
        }
        v34 = (_QWORD *)(*(_QWORD *)(a1 + 128) + 16 * v33);
        *v34 = v17;
        v35 = (unsigned __int64)v127;
        v34[1] = 0;
        v36 = *(_DWORD *)(a1 + 136);
        *(_DWORD *)(a1 + 136) = v36 + 1;
        v122 = (__int64 *)(v35 + 8LL * (unsigned int)v128);
        if ( (__int64 *)v35 == v122 )
        {
          v133 = v36;
          src = v136;
          v135 = 0x100000000LL;
          LODWORD(v132) = v123;
          HIDWORD(v132) = v36 - 1;
          goto LABEL_70;
        }
        v125 = (__int64 *)v35;
        v120 = a1 + 32;
        while ( 1 )
        {
          v37 = *(_DWORD *)(a1 + 56);
          v38 = *v125;
          if ( !v37 )
            break;
          LODWORD(a6) = v37 - 1;
          v39 = *(_QWORD *)(a1 + 40);
          v40 = (v37 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
          v41 = v39 + 16LL * v40;
          v42 = *(_QWORD *)v41;
          if ( v38 != *(_QWORD *)v41 )
          {
            v92 = 1;
            v93 = 0;
            while ( v42 != -8 )
            {
              if ( !v93 && v42 == -16 )
                v93 = v41;
              v40 = a6 & (v92 + v40);
              v41 = v39 + 16LL * v40;
              v42 = *(_QWORD *)v41;
              if ( v38 == *(_QWORD *)v41 )
                goto LABEL_32;
              ++v92;
            }
            v94 = *(_DWORD *)(a1 + 48);
            if ( v93 )
              v41 = v93;
            ++*(_QWORD *)(a1 + 32);
            v95 = v94 + 1;
            if ( 4 * v95 < 3 * v37 )
            {
              if ( v37 - *(_DWORD *)(a1 + 52) - v95 <= v37 >> 3 )
              {
                v121 = ((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4);
                sub_1F64680(v120, v37);
                v102 = *(_DWORD *)(a1 + 56);
                if ( !v102 )
                {
LABEL_185:
                  ++*(_DWORD *)(a1 + 48);
                  BUG();
                }
                v103 = v102 - 1;
                v104 = *(_QWORD *)(a1 + 40);
                a6 = 0;
                v105 = 1;
                LODWORD(v106) = v103 & v121;
                v95 = *(_DWORD *)(a1 + 48) + 1;
                v41 = v104 + 16LL * (v103 & v121);
                v107 = *(_QWORD *)v41;
                if ( v38 != *(_QWORD *)v41 )
                {
                  while ( v107 != -8 )
                  {
                    if ( v107 == -16 && !a6 )
                      a6 = v41;
                    v106 = v103 & (unsigned int)(v106 + v105);
                    v41 = v104 + 16 * v106;
                    v107 = *(_QWORD *)v41;
                    if ( v38 == *(_QWORD *)v41 )
                      goto LABEL_116;
                    ++v105;
                  }
                  goto LABEL_124;
                }
              }
              goto LABEL_116;
            }
LABEL_120:
            sub_1F64680(v120, 2 * v37);
            v96 = *(_DWORD *)(a1 + 56);
            if ( !v96 )
              goto LABEL_185;
            v97 = v96 - 1;
            v98 = *(_QWORD *)(a1 + 40);
            LODWORD(v99) = v97 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            v95 = *(_DWORD *)(a1 + 48) + 1;
            v41 = v98 + 16LL * (unsigned int)v99;
            v100 = *(_QWORD *)v41;
            if ( v38 != *(_QWORD *)v41 )
            {
              v101 = 1;
              a6 = 0;
              while ( v100 != -8 )
              {
                if ( v100 == -16 && !a6 )
                  a6 = v41;
                v99 = v97 & (unsigned int)(v99 + v101);
                v41 = v98 + 16 * v99;
                v100 = *(_QWORD *)v41;
                if ( v38 == *(_QWORD *)v41 )
                  goto LABEL_116;
                ++v101;
              }
LABEL_124:
              if ( a6 )
                v41 = a6;
            }
LABEL_116:
            *(_DWORD *)(a1 + 48) = v95;
            if ( *(_QWORD *)v41 != -8 )
              --*(_DWORD *)(a1 + 52);
            *(_QWORD *)v41 = v38;
            *(_DWORD *)(v41 + 8) = 0;
          }
LABEL_32:
          *(_DWORD *)(v41 + 8) = v36;
          v43 = *(_QWORD *)(v38 + 8);
          if ( v43 )
          {
            while ( 1 )
            {
              v44 = sub_1648700(v43);
              v45 = *((_BYTE *)v44 + 16);
              if ( v45 != 34 )
                goto LABEL_44;
              if ( (*((_BYTE *)v44 + 18) & 1) == 0 )
                break;
              v46 = (*((_BYTE *)v44 + 23) & 0x40) != 0
                  ? (_QWORD *)*(v44 - 1)
                  : &v44[-3 * (*((_DWORD *)v44 + 5) & 0xFFFFFFF)];
              v47 = v46[3];
              if ( !v47 )
                break;
              if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
              {
                v48 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
                    ? *(_QWORD *)(a2 - 8)
                    : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
                v49 = *(_QWORD *)(v48 + 24);
                if ( v47 == v49 )
                {
                  if ( v49 )
                    break;
                }
              }
LABEL_52:
              v43 = *(_QWORD *)(v43 + 8);
              if ( !v43 )
                goto LABEL_53;
            }
            sub_1F64840(a1, v44, v36);
            v45 = *((_BYTE *)v44 + 16);
LABEL_44:
            if ( v45 == 73 )
            {
              v50 = sub_1F5FF70((__int64)v44);
              if ( !v50
                || (*(_BYTE *)(a2 + 18) & 1) != 0
                && ((*(_BYTE *)(a2 + 23) & 0x40) == 0
                  ? (v51 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))
                  : (v51 = *(_QWORD *)(a2 - 8)),
                    (v52 = *(_QWORD *)(v51 + 24), v50 == v52) && v52) )
              {
                sub_1F64840(a1, v44, v36);
              }
            }
            goto LABEL_52;
          }
LABEL_53:
          if ( v122 == ++v125 )
          {
            v53 = v127;
            v54 = *(_DWORD *)(a1 + 136);
            src = v136;
            v135 = 0x100000000LL;
            v55 = v36 - 1;
            v56 = &v127[(unsigned int)v128];
            v57 = v123;
            HIDWORD(v132) = v36 - 1;
            LODWORD(v132) = v123;
            v133 = v54 - 1;
            if ( v127 != v56 )
            {
              do
              {
                v66 = *v53;
                v131.m128i_i64[1] = 0;
                v130.m128i_i64[1] = 0;
                v67 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                v68 = *(_QWORD *)(v66 - 24 * v67);
                if ( sub_1593BB0(v68, v57, 4 * v67, v55) )
                  v131.m128i_i64[0] = 0;
                else
                  v131.m128i_i64[0] = sub_1649C60(v68);
                v58 = *(_DWORD *)(v66 + 20) & 0xFFFFFFF;
                v59 = *(_QWORD *)(v66 + 24 * (1 - v58));
                v60 = *(_QWORD **)(v59 + 24);
                if ( *(_DWORD *)(v59 + 32) > 0x40u )
                  v60 = (_QWORD *)*v60;
                v130.m128i_i32[0] = (int)v60;
                v131.m128i_i64[1] = *(_QWORD *)(v66 + 40);
                v61 = sub_1649C60(*(_QWORD *)(v66 + 24 * (2 - v58)));
                if ( *(_BYTE *)(v61 + 16) == 53 )
                {
                  v130.m128i_i64[1] = v61;
                  v63 = (unsigned int)v135;
                  if ( (unsigned int)v135 >= HIDWORD(v135) )
                  {
LABEL_68:
                    v57 = (__int64)v136;
                    sub_16CD150((__int64)&src, v136, 0, 32, v62, a6);
                    v63 = (unsigned int)v135;
                  }
                }
                else
                {
                  v63 = (unsigned int)v135;
                  if ( (unsigned int)v135 >= HIDWORD(v135) )
                    goto LABEL_68;
                }
                v64 = _mm_load_si128(&v131);
                ++v53;
                v65 = (__m128i *)((char *)src + 32 * v63);
                *v65 = _mm_load_si128(&v130);
                v65[1] = v64;
                LODWORD(v135) = v135 + 1;
              }
              while ( v56 != v53 );
            }
LABEL_70:
            v69 = *(_DWORD *)(a1 + 216);
            if ( v69 >= *(_DWORD *)(a1 + 220) )
            {
              sub_1F60AD0((unsigned __int64 *)(a1 + 208), 0);
              v69 = *(_DWORD *)(a1 + 216);
            }
            v70 = *(_QWORD *)(a1 + 208) + ((unsigned __int64)v69 << 6);
            if ( v70 )
            {
              v71 = (void *)(v70 + 32);
              v72 = (void **)(v70 + 16);
              *(_QWORD *)v70 = v132;
              v73 = v133;
              *(_QWORD *)(v70 + 16) = v70 + 32;
              *(_DWORD *)(v70 + 8) = v73;
              *(_QWORD *)(v70 + 24) = 0x100000000LL;
              v74 = v135;
              if ( !(_DWORD)v135 || v72 == &src )
              {
                v75 = src;
                v69 = *(_DWORD *)(a1 + 216);
              }
              else
              {
                if ( (_DWORD)v135 == 1 )
                {
                  v75 = src;
                  v76 = 32;
                  goto LABEL_77;
                }
                sub_16CD150((__int64)v72, (const void *)(v70 + 32), (unsigned int)v135, 32, (int)v71, a6);
                v71 = *(void **)(v70 + 16);
                v75 = src;
                v76 = 32LL * (unsigned int)v135;
                if ( v76 )
                {
LABEL_77:
                  memcpy(v71, v75, v76);
                  v75 = src;
                }
                *(_DWORD *)(v70 + 24) = v74;
                v69 = *(_DWORD *)(a1 + 216);
              }
            }
            else
            {
              v75 = src;
            }
            *(_DWORD *)(a1 + 216) = v69 + 1;
            if ( v75 != v136 )
              _libc_free((unsigned __int64)v75);
            if ( v127 != (__int64 *)v129 )
              _libc_free((unsigned __int64)v127);
            return;
          }
        }
        ++*(_QWORD *)(a1 + 32);
        goto LABEL_120;
      }
      v108 = 1;
      a6 = 0;
      while ( v26 != -8 )
      {
        if ( !a6 && v26 == -16 )
          a6 = v25;
        v24 = v23 & (v108 + v24);
        v25 = v22 + 16LL * v24;
        v26 = *(_QWORD *)v25;
        if ( a2 == *(_QWORD *)v25 )
          goto LABEL_14;
        ++v108;
      }
      if ( a6 )
        v25 = a6;
      ++*(_QWORD *)a1;
      v109 = *(_DWORD *)(a1 + 16) + 1;
      if ( 4 * v109 < 3 * v21 )
      {
        if ( v21 - *(_DWORD *)(a1 + 20) - v109 > v21 >> 3 )
        {
LABEL_141:
          *(_DWORD *)(a1 + 16) = v109;
          if ( *(_QWORD *)v25 != -8 )
            --*(_DWORD *)(a1 + 20);
          *(_QWORD *)v25 = a2;
          *(_DWORD *)(v25 + 8) = 0;
          goto LABEL_14;
        }
        sub_1F61920(a1, v21);
        v115 = *(_DWORD *)(a1 + 24);
        if ( v115 )
        {
          v116 = v115 - 1;
          a6 = *(_QWORD *)(a1 + 8);
          v117 = 1;
          LODWORD(v118) = v116 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v109 = *(_DWORD *)(a1 + 16) + 1;
          v119 = 0;
          v25 = a6 + 16LL * (unsigned int)v118;
          v23 = *(_QWORD *)v25;
          if ( a2 != *(_QWORD *)v25 )
          {
            while ( v23 != -8 )
            {
              if ( v23 == -16 && !v119 )
                v119 = v25;
              v118 = v116 & (unsigned int)(v118 + v117);
              v25 = a6 + 16 * v118;
              v23 = *(_QWORD *)v25;
              if ( a2 == *(_QWORD *)v25 )
                goto LABEL_141;
              ++v117;
            }
            if ( v119 )
              v25 = v119;
          }
          goto LABEL_141;
        }
LABEL_186:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_1F61920(a1, 2 * v21);
    v110 = *(_DWORD *)(a1 + 24);
    if ( v110 )
    {
      v111 = v110 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      LODWORD(v112) = v111 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v109 = *(_DWORD *)(a1 + 16) + 1;
      v25 = v23 + 16LL * (unsigned int)v112;
      v113 = *(_QWORD *)v25;
      if ( a2 != *(_QWORD *)v25 )
      {
        v114 = 1;
        a6 = 0;
        while ( v113 != -8 )
        {
          if ( !a6 && v113 == -16 )
            a6 = v25;
          v112 = v111 & (unsigned int)(v112 + v114);
          v25 = v23 + 16 * v112;
          v113 = *(_QWORD *)v25;
          if ( a2 == *(_QWORD *)v25 )
            goto LABEL_141;
          ++v114;
        }
        if ( a6 )
          v25 = a6;
      }
      goto LABEL_141;
    }
    goto LABEL_186;
  }
  v77 = *(_DWORD *)(a1 + 24);
  if ( v77 )
  {
    v78 = v77 - 1;
    v79 = *(_QWORD *)(a1 + 8);
    v80 = (v77 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v81 = *(_QWORD *)(v79 + 16LL * v80);
    if ( a2 == v81 )
      return;
    v82 = 1;
    while ( v81 != -8 )
    {
      a5 = v82 + 1;
      v80 = v78 & (v82 + v80);
      v81 = *(_QWORD *)(v79 + 16LL * v80);
      if ( a2 == v81 )
        return;
      ++v82;
    }
  }
  v83 = *(unsigned int *)(a1 + 136);
  if ( (unsigned int)v83 >= *(_DWORD *)(a1 + 140) )
  {
    sub_16CD150(a1 + 128, (const void *)(a1 + 144), 0, 16, a5, a6);
    v83 = *(unsigned int *)(a1 + 136);
  }
  v84 = (_QWORD *)(*(_QWORD *)(a1 + 128) + 16 * v83);
  *v84 = v6;
  v84[1] = v8;
  v85 = *(_DWORD *)(a1 + 136);
  v132 = a2;
  *(_DWORD *)(a1 + 136) = v85 + 1;
  *((_DWORD *)sub_1F61AE0(a1, &v132) + 2) = v85;
  v86 = *(_QWORD *)(v8 + 8);
  if ( v86 )
  {
    while ( 1 )
    {
      v87 = sub_1648700(v86);
      if ( (unsigned __int8)(*((_BYTE *)v87 + 16) - 25) <= 9u )
        break;
      v86 = *(_QWORD *)(v86 + 8);
      if ( !v86 )
        goto LABEL_102;
    }
LABEL_99:
    v88 = sub_1F60340(v87[5], *(_QWORD *)(a2 - 24));
    if ( v88 )
    {
      v89 = sub_157ED20(v88);
      sub_1F64840(a1, v89, v85);
    }
    while ( 1 )
    {
      v86 = *(_QWORD *)(v86 + 8);
      if ( !v86 )
        break;
      v87 = sub_1648700(v86);
      if ( (unsigned __int8)(*((_BYTE *)v87 + 16) - 25) <= 9u )
        goto LABEL_99;
    }
  }
LABEL_102:
  for ( j = *(_QWORD *)(a2 + 8); j; j = *(_QWORD *)(j + 8) )
  {
    v91 = *((unsigned __int8 *)sub_1648700(j) + 16) - 34;
    if ( v91 <= 0x36 && ((1LL << v91) & 0x40018000000001LL) != 0 )
      sub_16BD130("Cleanup funclets for the MSVC++ personality cannot contain exceptional actions", 1u);
  }
}
