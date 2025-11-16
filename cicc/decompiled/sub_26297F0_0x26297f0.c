// Function: sub_26297F0
// Address: 0x26297f0
//
__int64 *__fastcall sub_26297F0(__int64 a1, __int64 *a2, __int64 a3, unsigned __int8 *a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // r14
  _QWORD *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // r15
  unsigned __int64 v25; // rdi
  unsigned int v26; // esi
  __int64 v27; // r13
  __int64 v28; // r8
  int v29; // r15d
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 **v34; // r13
  __int64 v35; // rax
  size_t v36; // rdx
  size_t v37; // rdi
  _QWORD *v38; // rdx
  size_t v39; // rax
  _QWORD *v40; // rdi
  __int64 v41; // r14
  int v42; // eax
  __int64 v43; // r14
  __int64 *v44; // r12
  __int64 *j; // r13
  __int64 v46; // rsi
  _QWORD *v47; // r15
  int *v48; // r12
  unsigned __int64 v49; // rdi
  __int64 *v50; // rdx
  __int64 *v51; // r15
  __int64 v52; // rax
  __int64 *v53; // r12
  __int64 *v54; // r13
  __int64 v55; // r14
  _QWORD *v56; // rdx
  __int64 v57; // rdx
  unsigned __int8 v58; // al
  __int64 v59; // rdx
  _QWORD *v60; // rax
  char *v61; // rax
  __int64 v62; // rdx
  unsigned __int64 v63; // r8
  __int64 v64; // rdi
  __int64 v65; // r12
  __int64 v66; // rdi
  __int64 v67; // rsi
  _QWORD *v68; // rax
  _QWORD *v69; // rax
  _QWORD *v70; // rax
  int v71; // edx
  unsigned __int64 v72; // rax
  int v73; // edx
  __int64 v74; // rdx
  unsigned __int64 v75; // rcx
  unsigned __int64 v76; // rax
  bool v77; // cf
  unsigned __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // r15
  __int64 i; // r14
  __int64 v82; // rax
  __int64 v83; // rdx
  int v84; // esi
  unsigned __int64 v85; // r13
  unsigned __int64 v86; // rdi
  __int64 v87; // rax
  int v88; // r12d
  int v89; // r12d
  unsigned int v90; // ecx
  __int64 v91; // r8
  int v92; // edi
  __int64 v93; // rsi
  __int64 v94; // rax
  __int64 v95; // rax
  int v96; // r11d
  int v97; // r11d
  __int64 v98; // r8
  __int64 v99; // rcx
  unsigned int v100; // r14d
  int v101; // esi
  __int64 v102; // rdi
  _QWORD *v103; // rax
  __int64 v104; // rax
  unsigned __int64 v105; // r14
  __int64 v106; // [rsp+0h] [rbp-210h]
  unsigned __int64 v107; // [rsp+8h] [rbp-208h]
  unsigned __int64 v108; // [rsp+10h] [rbp-200h]
  __int64 v109; // [rsp+18h] [rbp-1F8h]
  _QWORD *v110; // [rsp+20h] [rbp-1F0h]
  __int64 *v112; // [rsp+38h] [rbp-1D8h]
  __int64 *v114; // [rsp+58h] [rbp-1B8h]
  _QWORD *v115; // [rsp+68h] [rbp-1A8h]
  char *v116; // [rsp+68h] [rbp-1A8h]
  __int64 v118; // [rsp+88h] [rbp-188h] BYREF
  _QWORD *v119; // [rsp+90h] [rbp-180h] BYREF
  size_t v120; // [rsp+98h] [rbp-178h]
  __int64 v121; // [rsp+A0h] [rbp-170h] BYREF
  unsigned int v122; // [rsp+A8h] [rbp-168h]
  char v123; // [rsp+B0h] [rbp-160h]
  _QWORD *v124; // [rsp+C0h] [rbp-150h] BYREF
  const void **v125; // [rsp+C8h] [rbp-148h]
  __int64 v126; // [rsp+D0h] [rbp-140h] BYREF
  unsigned int v127; // [rsp+D8h] [rbp-138h]
  __int16 v128; // [rsp+E0h] [rbp-130h]
  char v129[8]; // [rsp+F0h] [rbp-120h] BYREF
  char v130; // [rsp+F8h] [rbp-118h] BYREF
  int *v131; // [rsp+100h] [rbp-110h]
  char *v132; // [rsp+108h] [rbp-108h]
  __int64 v133; // [rsp+118h] [rbp-F8h]
  __int64 v134; // [rsp+120h] [rbp-F0h]
  unsigned __int64 v135; // [rsp+128h] [rbp-E8h]
  unsigned int v136; // [rsp+130h] [rbp-E0h]
  unsigned __int64 *v137; // [rsp+140h] [rbp-D0h] BYREF
  __int64 v138; // [rsp+148h] [rbp-C8h]
  unsigned __int64 v139; // [rsp+150h] [rbp-C0h] BYREF
  unsigned __int64 v140; // [rsp+158h] [rbp-B8h]
  _QWORD *v141; // [rsp+160h] [rbp-B0h]
  __int64 v142; // [rsp+168h] [rbp-A8h]
  unsigned __int64 v143; // [rsp+170h] [rbp-A0h]
  unsigned __int64 v144; // [rsp+1D0h] [rbp-40h]
  char *v145; // [rsp+1D8h] [rbp-38h]

  result = a2;
  v112 = &a2[a3];
  if ( v112 == a2 )
    return result;
  v114 = a2;
  do
  {
    v144 = -1;
    v145 = 0;
    v7 = *v114;
    v137 = &v139;
    v138 = 0x1000000000LL;
    if ( *(_DWORD *)(a5 + 16) )
    {
      v50 = *(__int64 **)(a5 + 8);
      v51 = &v50[2 * *(unsigned int *)(a5 + 24)];
      if ( v50 != v51 )
      {
        while ( 1 )
        {
          v52 = *v50;
          v53 = v50;
          if ( *v50 != -4096 && v52 != -8192 )
            break;
          v50 += 2;
          if ( v51 == v50 )
            goto LABEL_4;
        }
        while ( 1 )
        {
          if ( v51 == v53 )
            goto LABEL_4;
          v54 = (__int64 *)(v52 + 24);
          v55 = v52 + 24 + 8LL * *(_QWORD *)(v52 + 8);
          if ( v52 + 24 != v55 )
            break;
LABEL_89:
          v53 += 2;
          if ( v53 == v51 )
            goto LABEL_4;
          while ( 1 )
          {
            v52 = *v53;
            if ( *v53 != -8192 && v52 != -4096 )
              break;
            v53 += 2;
            if ( v51 == v53 )
              goto LABEL_4;
          }
        }
        while ( 1 )
        {
          v57 = *v54;
          v58 = *(_BYTE *)(*v54 - 16);
          if ( (v58 & 2) != 0 )
          {
            v56 = *(_QWORD **)(v57 - 32);
            if ( v7 != v56[1] )
              goto LABEL_77;
LABEL_80:
            v59 = *(_QWORD *)(*v56 + 136LL);
            v60 = *(_QWORD **)(v59 + 24);
            if ( *(_DWORD *)(v59 + 32) > 0x40u )
              v60 = (_QWORD *)*v60;
            v61 = (char *)v60 + v53[1];
            if ( (unsigned __int64)v61 < v144 )
              v144 = (unsigned __int64)v61;
            if ( v61 > v145 )
              v145 = v61;
            v62 = (unsigned int)v138;
            v63 = (unsigned int)v138 + 1LL;
            if ( v63 > HIDWORD(v138) )
            {
              v116 = v61;
              sub_C8D5F0((__int64)&v137, &v139, (unsigned int)v138 + 1LL, 8u, v63, a6);
              v62 = (unsigned int)v138;
              v61 = v116;
            }
            ++v54;
            v137[v62] = (unsigned __int64)v61;
            LODWORD(v138) = v138 + 1;
            if ( (__int64 *)v55 == v54 )
              goto LABEL_89;
          }
          else
          {
            v56 = (_QWORD *)(-16 - 8LL * ((v58 >> 2) & 0xF) + v57);
            if ( v7 == v56[1] )
              goto LABEL_80;
LABEL_77:
            if ( (__int64 *)v55 == ++v54 )
              goto LABEL_89;
          }
        }
      }
    }
LABEL_4:
    sub_2629680((__int64)v129, (__int64)&v137);
    if ( v137 != &v139 )
      _libc_free((unsigned __int64)v137);
    LODWORD(v137) = 0;
    v123 = 0;
    v8 = sub_ACD640(*(_QWORD *)(a1 + 112), v134, 0);
    LOBYTE(v128) = 0;
    v9 = *(_QWORD *)(a1 + 64);
    v10 = v8;
    if ( v123 )
    {
      LODWORD(v125) = v120;
      if ( (unsigned int)v120 > 0x40 )
        sub_C43780((__int64)&v124, (const void **)&v119);
      else
        v124 = v119;
      v127 = v122;
      if ( v122 > 0x40 )
        sub_C43780((__int64)&v126, (const void **)&v121);
      else
        v126 = v121;
      LOBYTE(v128) = 1;
    }
    v118 = v10;
    v11 = sub_AD9FD0(v9, a4, &v118, 1, 0, (__int64)&v124, 0);
    if ( (_BYTE)v128 )
    {
      LOBYTE(v128) = 0;
      if ( v127 > 0x40 && v126 )
        j_j___libc_free_0_0(v126);
      if ( (unsigned int)v125 > 0x40 && v124 )
        j_j___libc_free_0_0((unsigned __int64)v124);
    }
    v138 = v11;
    v139 = sub_ACD640(*(_QWORD *)(a1 + 64), v136, 0);
    if ( v123 )
    {
      v123 = 0;
      if ( v122 > 0x40 && v121 )
        j_j___libc_free_0_0(v121);
      if ( (unsigned int)v120 > 0x40 && v119 )
        j_j___libc_free_0_0((unsigned __int64)v119);
    }
    v12 = sub_ACD640(*(_QWORD *)(a1 + 112), v135 - 1, 0);
    v13 = v135;
    v140 = v12;
    if ( v135 == v133 )
    {
      v14 = 0;
      LODWORD(v137) = (v135 != 1) + 3;
      goto LABEL_32;
    }
    if ( v135 <= 0x40 )
    {
      v64 = (__int64)v132;
      LODWORD(v137) = 2;
      if ( v132 == &v130 )
        goto LABEL_107;
      v65 = 0;
      do
      {
        v65 |= 1LL << *(_QWORD *)(v64 + 32);
        v64 = sub_220EF30(v64);
      }
      while ( (char *)v64 != &v130 );
      if ( !v65 )
      {
LABEL_107:
        LODWORD(v137) = 0;
        v14 = 0;
      }
      else
      {
        if ( v13 > 0x20 )
          v66 = *(_QWORD *)(a1 + 104);
        else
          v66 = *(_QWORD *)(a1 + 88);
        v67 = v65;
        v14 = 0;
        v143 = sub_ACD640(v66, v67, 0);
      }
      goto LABEL_32;
    }
    LODWORD(v137) = 1;
    v128 = 257;
    BYTE4(v119) = 0;
    v115 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v115 )
      sub_B30000((__int64)v115, *(_QWORD *)a1, *(_QWORD **)(a1 + 64), 1, 8, 0, (__int64)&v124, 0, 0, (__int64)v119, 0);
    BYTE4(v119) = 0;
    v128 = 257;
    v110 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v110 )
      sub_B30000((__int64)v110, *(_QWORD *)a1, *(_QWORD **)(a1 + 64), 1, 8, 0, (__int64)&v124, 0, 0, (__int64)v119, 0);
    v14 = *(_QWORD *)(a1 + 168);
    if ( v14 == *(_QWORD *)(a1 + 176) )
    {
      v74 = v14 - *(_QWORD *)(a1 + 160);
      v108 = *(_QWORD *)(a1 + 160);
      v75 = 0xCCCCCCCCCCCCCCCDLL * (v74 >> 4);
      if ( v75 == 0x199999999999999LL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v76 = 1;
      if ( v75 )
        v76 = 0xCCCCCCCCCCCCCCCDLL * (v74 >> 4);
      v77 = __CFADD__(v75, v76);
      v78 = v75 + v76;
      if ( v77 )
      {
        v105 = 0x7FFFFFFFFFFFFFD0LL;
      }
      else
      {
        if ( !v78 )
        {
          v107 = 0;
          v15 = 80;
          v79 = v14 - *(_QWORD *)(a1 + 160);
          v109 = 0;
          if ( !v74 )
          {
LABEL_145:
            v80 = v108;
            if ( v14 == v108 )
            {
              v14 = v109;
            }
            else
            {
              for ( i = v109; ; i += 80 )
              {
                if ( i )
                {
                  v82 = *(_QWORD *)(v80 + 16);
                  v83 = i + 8;
                  if ( v82 )
                  {
                    v84 = *(_DWORD *)(v80 + 8);
                    *(_QWORD *)(i + 16) = v82;
                    *(_DWORD *)(i + 8) = v84;
                    *(_QWORD *)(i + 24) = *(_QWORD *)(v80 + 24);
                    *(_QWORD *)(i + 32) = *(_QWORD *)(v80 + 32);
                    *(_QWORD *)(v82 + 8) = v83;
                    *(_QWORD *)(i + 40) = *(_QWORD *)(v80 + 40);
                    *(_QWORD *)(v80 + 16) = 0;
                    *(_QWORD *)(v80 + 24) = v80 + 8;
                    *(_QWORD *)(v80 + 32) = v80 + 8;
                    *(_QWORD *)(v80 + 40) = 0;
                  }
                  else
                  {
                    *(_DWORD *)(i + 8) = 0;
                    *(_QWORD *)(i + 16) = 0;
                    *(_QWORD *)(i + 24) = v83;
                    *(_QWORD *)(i + 32) = v83;
                    *(_QWORD *)(i + 40) = 0;
                  }
                  *(_QWORD *)(i + 48) = *(_QWORD *)(v80 + 48);
                  *(_QWORD *)(i + 56) = *(_QWORD *)(v80 + 56);
                  *(_QWORD *)(i + 64) = *(_QWORD *)(v80 + 64);
                  *(_QWORD *)(i + 72) = *(_QWORD *)(v80 + 72);
                }
                v85 = *(_QWORD *)(v80 + 16);
                while ( v85 )
                {
                  sub_261DCB0(*(_QWORD *)(v85 + 24));
                  v86 = v85;
                  v85 = *(_QWORD *)(v85 + 16);
                  j_j___libc_free_0(v86);
                }
                v80 += 80LL;
                v87 = i + 80;
                if ( v14 == v80 )
                  break;
              }
              v15 = i + 160;
              v14 = v87;
            }
            if ( v108 )
              j_j___libc_free_0(v108);
            *(_QWORD *)(a1 + 160) = v109;
            *(_QWORD *)(a1 + 168) = v15;
            *(_QWORD *)(a1 + 176) = v107;
            goto LABEL_19;
          }
LABEL_144:
          *(_QWORD *)(v79 + 40) = 0;
          *(_OWORD *)(v79 + 16) = 0;
          *(_QWORD *)(v79 + 32) = v79 + 8;
          *(_QWORD *)(v79 + 24) = v79 + 8;
          *(_OWORD *)v79 = 0;
          *(_OWORD *)(v79 + 48) = 0;
          *(_OWORD *)(v79 + 64) = 0;
          goto LABEL_145;
        }
        if ( v78 > 0x199999999999999LL )
          v78 = 0x199999999999999LL;
        v105 = 80 * v78;
      }
      v106 = v14 - *(_QWORD *)(a1 + 160);
      v109 = sub_22077B0(v105);
      v107 = v109 + v105;
      v15 = v109 + 80;
      v79 = v109 + v106;
      if ( !(v109 + v106) )
        goto LABEL_145;
      goto LABEL_144;
    }
    if ( v14 )
    {
      *(_QWORD *)(v14 + 40) = 0;
      *(_OWORD *)(v14 + 16) = 0;
      *(_QWORD *)(v14 + 32) = v14 + 8;
      *(_QWORD *)(v14 + 24) = v14 + 8;
      *(_OWORD *)v14 = 0;
      *(_OWORD *)(v14 + 48) = 0;
      *(_OWORD *)(v14 + 64) = 0;
      v14 = *(_QWORD *)(a1 + 168);
    }
    v15 = v14 + 80;
    *(_QWORD *)(a1 + 168) = v14 + 80;
LABEL_19:
    if ( (char *)(v15 - 80) != v129 )
    {
      v16 = *(_QWORD **)(v15 - 64);
      v124 = v16;
      v17 = *(_QWORD *)(v15 - 48);
      v126 = v15 - 80;
      v125 = (const void **)v17;
      if ( v16 )
      {
        v16[1] = 0;
        if ( *(_QWORD *)(v17 + 16) )
          v125 = *(const void ***)(v17 + 16);
      }
      else
      {
        v125 = 0;
      }
      *(_QWORD *)(v15 - 64) = 0;
      *(_QWORD *)(v15 - 56) = v15 - 72;
      *(_QWORD *)(v15 - 48) = v15 - 72;
      *(_QWORD *)(v15 - 40) = 0;
      if ( v131 )
      {
        v18 = sub_261A730(v131, v15 - 72, &v124);
        v19 = v18;
        do
        {
          v20 = v18;
          v18 = *(_QWORD *)(v18 + 16);
        }
        while ( v18 );
        *(_QWORD *)(v15 - 56) = v20;
        v21 = v19;
        do
        {
          v22 = v21;
          v21 = *(_QWORD *)(v21 + 24);
        }
        while ( v21 );
        *(_QWORD *)(v15 - 48) = v22;
        v23 = v133;
        *(_QWORD *)(v15 - 64) = v19;
        *(_QWORD *)(v15 - 40) = v23;
      }
      v24 = v124;
      while ( v24 )
      {
        sub_261DCB0(v24[3]);
        v25 = (unsigned __int64)v24;
        v24 = (_QWORD *)v24[2];
        j_j___libc_free_0(v25);
      }
    }
    *(_QWORD *)(v15 - 32) = v135;
    *(_QWORD *)(v15 - 16) = v110;
    *(_QWORD *)(v15 - 24) = v115;
    v141 = v115;
    v142 = *(_QWORD *)(v15 - 16);
LABEL_32:
    v26 = *(_DWORD *)(a1 + 152);
    v27 = a1 + 128;
    if ( !v26 )
    {
      ++*(_QWORD *)(a1 + 128);
      goto LABEL_165;
    }
    v28 = *(_QWORD *)(a1 + 136);
    a6 = v26 - 1;
    v29 = 1;
    LODWORD(v30) = a6 & (((unsigned int)v7 >> 4) ^ ((unsigned int)v7 >> 9));
    v31 = v28 + 40LL * (unsigned int)v30;
    v32 = 0;
    v33 = *(_QWORD *)v31;
    if ( v7 == *(_QWORD *)v31 )
    {
LABEL_34:
      v34 = (__int64 **)(v31 + 8);
      if ( !*(_BYTE *)(v31 + 32) )
        goto LABEL_42;
      v35 = sub_B91420(v7);
      v37 = v36;
      v38 = (_QWORD *)v35;
      v120 = v37;
      v39 = v37;
      v40 = *(_QWORD **)(a1 + 8);
      v119 = v38;
      v41 = sub_9CA790(v40, v38, v39);
      v42 = (int)v137;
      *(_DWORD *)v41 = (_DWORD)v137;
      v124 = (_QWORD *)a1;
      v125 = (const void **)&v119;
      if ( v42 )
        sub_2623820((__int64)&v124, (__int64)"global_addr", 11, v138);
      if ( (unsigned int)((_DWORD)v137 - 1) > 1 && (_DWORD)v137 != 4 )
        goto LABEL_39;
      if ( (unsigned int)(*(_DWORD *)(a1 + 28) - 38) <= 1 && *(_DWORD *)(a1 + 36) == 3 )
      {
        v95 = sub_AD4C70(v139, *(__int64 ***)(a1 + 72), 0);
        sub_2623820((__int64)&v124, (__int64)"align", 5, v95);
      }
      else
      {
        v68 = *(_QWORD **)(v139 + 24);
        if ( *(_DWORD *)(v139 + 32) > 0x40u )
          v68 = (_QWORD *)*v68;
        *(_QWORD *)(v41 + 8) = v68;
      }
      if ( (unsigned int)(*(_DWORD *)(a1 + 28) - 38) <= 1 && *(_DWORD *)(a1 + 36) == 3 )
      {
        v94 = sub_AD4C70(v140, *(__int64 ***)(a1 + 72), 0);
        sub_2623820((__int64)&v124, (__int64)"size_m1", 7, v94);
      }
      else
      {
        v69 = *(_QWORD **)(v140 + 24);
        if ( *(_DWORD *)(v140 + 32) > 0x40u )
          v69 = (_QWORD *)*v69;
        *(_QWORD *)(v41 + 16) = v69;
      }
      v70 = *(_QWORD **)(v140 + 24);
      if ( *(_DWORD *)(v140 + 32) > 0x40u )
        v70 = (_QWORD *)*v70;
      v71 = (int)v137;
      v72 = (unsigned __int64)v70 + 1;
      if ( (_DWORD)v137 == 2 )
      {
        *(_DWORD *)(v41 + 4) = (v72 > 0x20) + 5;
      }
      else
      {
        *(_DWORD *)(v41 + 4) = v72 < 0x81 ? 7 : 32;
        if ( v71 != 1 )
          goto LABEL_39;
        sub_2623820((__int64)&v124, (__int64)"byte_array", 10, (__int64)v141);
        if ( (unsigned int)(*(_DWORD *)(a1 + 28) - 38) > 1 || *(_DWORD *)(a1 + 36) != 3 )
        {
          v43 = v41 + 24;
LABEL_40:
          if ( v14 )
            *(_QWORD *)(v14 + 72) = v43;
          goto LABEL_42;
        }
        sub_2623820((__int64)&v124, (__int64)"bit_mask", 8, v142);
        if ( (_DWORD)v137 != 2 )
        {
LABEL_39:
          v43 = 0;
          goto LABEL_40;
        }
      }
      if ( (unsigned int)(*(_DWORD *)(a1 + 28) - 38) <= 1 && *(_DWORD *)(a1 + 36) == 3 )
      {
        v104 = sub_AD4C70(v143, *(__int64 ***)(a1 + 72), 0);
        sub_2623820((__int64)&v124, (__int64)"inline_bits", 11, v104);
      }
      else
      {
        v103 = *(_QWORD **)(v143 + 24);
        if ( *(_DWORD *)(v143 + 32) > 0x40u )
          v103 = (_QWORD *)*v103;
        *(_QWORD *)(v41 + 32) = v103;
      }
      goto LABEL_39;
    }
    while ( v33 != -4096 )
    {
      if ( !v32 && v33 == -8192 )
        v32 = v31;
      v30 = (unsigned int)a6 & ((_DWORD)v30 + v29);
      v31 = v28 + 40 * v30;
      v33 = *(_QWORD *)v31;
      if ( v7 == *(_QWORD *)v31 )
        goto LABEL_34;
      ++v29;
    }
    if ( !v32 )
      v32 = v31;
    ++*(_QWORD *)(a1 + 128);
    v73 = *(_DWORD *)(a1 + 144) + 1;
    if ( 4 * v73 >= 3 * v26 )
    {
LABEL_165:
      sub_261CF40(v27, 2 * v26);
      v88 = *(_DWORD *)(a1 + 152);
      if ( v88 )
      {
        v89 = v88 - 1;
        a6 = *(_QWORD *)(a1 + 136);
        v90 = v89 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v73 = *(_DWORD *)(a1 + 144) + 1;
        v32 = a6 + 40LL * v90;
        v91 = *(_QWORD *)v32;
        if ( v7 != *(_QWORD *)v32 )
        {
          v92 = 1;
          v93 = 0;
          while ( v91 != -4096 )
          {
            if ( !v93 && v91 == -8192 )
              v93 = v32;
            v90 = v89 & (v92 + v90);
            v32 = a6 + 40LL * v90;
            v91 = *(_QWORD *)v32;
            if ( v7 == *(_QWORD *)v32 )
              goto LABEL_135;
            ++v92;
          }
          if ( v93 )
            v32 = v93;
        }
        goto LABEL_135;
      }
      goto LABEL_207;
    }
    if ( v26 - *(_DWORD *)(a1 + 148) - v73 <= v26 >> 3 )
    {
      sub_261CF40(v27, v26);
      v96 = *(_DWORD *)(a1 + 152);
      if ( v96 )
      {
        v97 = v96 - 1;
        v98 = *(_QWORD *)(a1 + 136);
        v99 = 0;
        v100 = v97 & (((unsigned int)v7 >> 4) ^ ((unsigned int)v7 >> 9));
        v73 = *(_DWORD *)(a1 + 144) + 1;
        v101 = 1;
        v32 = v98 + 40LL * v100;
        v102 = *(_QWORD *)v32;
        if ( v7 != *(_QWORD *)v32 )
        {
          while ( v102 != -4096 )
          {
            if ( !v99 && v102 == -8192 )
              v99 = v32;
            a6 = (unsigned int)(v101 + 1);
            v100 = v97 & (v101 + v100);
            v32 = v98 + 40LL * v100;
            v102 = *(_QWORD *)v32;
            if ( v7 == *(_QWORD *)v32 )
              goto LABEL_135;
            ++v101;
          }
          if ( v99 )
            v32 = v99;
        }
        goto LABEL_135;
      }
LABEL_207:
      ++*(_DWORD *)(a1 + 144);
      BUG();
    }
LABEL_135:
    *(_DWORD *)(a1 + 144) = v73;
    if ( *(_QWORD *)v32 != -4096 )
      --*(_DWORD *)(a1 + 148);
    *(_QWORD *)v32 = v7;
    v34 = (__int64 **)(v32 + 8);
    *(_OWORD *)(v32 + 8) = 0;
    *(_OWORD *)(v32 + 24) = 0;
LABEL_42:
    v44 = v34[1];
    for ( j = *v34; v44 != j; ++j )
    {
      v47 = (_QWORD *)*j;
      if ( (_DWORD)v137 != 5 )
      {
        v46 = (_DWORD)v137 ? sub_26267C0(a1, v7, *j, (__int64)&v137) : sub_ACD720(**(__int64 ***)a1);
        if ( v46 )
        {
          sub_BD84D0((__int64)v47, v46);
          sub_B43D60(v47);
        }
      }
    }
    v48 = v131;
    while ( v48 )
    {
      sub_261DCB0(*((_QWORD *)v48 + 3));
      v49 = (unsigned __int64)v48;
      v48 = (int *)*((_QWORD *)v48 + 2);
      j_j___libc_free_0(v49);
    }
    result = ++v114;
  }
  while ( v112 != v114 );
  return result;
}
