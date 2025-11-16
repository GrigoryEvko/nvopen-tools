// Function: sub_1BE0300
// Address: 0x1be0300
//
__int64 __fastcall sub_1BE0300(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        __m128 a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 v15; // r15
  char *v16; // rax
  __int64 v17; // r13
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 *v20; // r14
  __int64 *v21; // r13
  __int64 v22; // rbx
  char *v23; // rax
  char *v24; // r15
  __int64 *v25; // rdi
  __int64 v26; // r10
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 ***v29; // r13
  __int64 ***v30; // rbx
  __int64 ***v31; // r15
  __int64 *v32; // r8
  __int64 *v33; // r11
  __int64 **v34; // rsi
  __int64 ***v35; // rdi
  __int64 ***v36; // rax
  __int64 ***v37; // rcx
  __int64 v38; // rbx
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 *v41; // rax
  __int64 v42; // rbx
  __int64 v43; // r15
  char v44; // dl
  int v45; // eax
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // rax
  __int64 *v49; // rdx
  _QWORD *v50; // rax
  __int64 v51; // r13
  int v52; // edx
  __int64 v53; // rsi
  int v54; // edx
  int v55; // ecx
  __int64 *v56; // rax
  __int64 v57; // r9
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 *v61; // rcx
  _QWORD *v62; // rax
  unsigned __int8 v63; // al
  __int64 v64; // r12
  __int64 *v65; // rax
  __int64 *v66; // rdi
  __int64 *v67; // rdx
  _QWORD *v68; // rdi
  int v69; // edx
  __int64 v70; // rax
  __int64 *v71; // rbx
  __int64 *v72; // r13
  __int64 v73; // rax
  unsigned __int64 v74; // rax
  __int64 v75; // rdx
  int v76; // eax
  unsigned __int64 *v77; // rdi
  __int64 v78; // rax
  bool v79; // zf
  _QWORD *v80; // rbx
  _QWORD *v81; // r12
  __int64 v82; // rax
  int v84; // r8d
  double v85; // xmm4_8
  double v86; // xmm5_8
  char v87; // r13
  unsigned __int8 v88; // r13
  bool v89; // al
  int v90; // esi
  unsigned int v91; // eax
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 v94; // r11
  __int64 *v95; // r10
  __int64 v96; // rcx
  __int64 v97; // r15
  __int64 *v98; // rbx
  __int64 *v99; // r12
  __int64 v100; // rdx
  char v101; // al
  char *v102; // rdx
  int v103; // eax
  int v104; // edi
  __int64 v105; // rax
  __int64 v106; // [rsp+8h] [rbp-2A8h]
  __int64 v107; // [rsp+10h] [rbp-2A0h]
  __int64 v108; // [rsp+10h] [rbp-2A0h]
  __int64 v109; // [rsp+10h] [rbp-2A0h]
  unsigned int v110; // [rsp+28h] [rbp-288h]
  __int64 *v112; // [rsp+38h] [rbp-278h]
  __int64 v113; // [rsp+38h] [rbp-278h]
  __int64 v114; // [rsp+40h] [rbp-270h]
  __int64 v115; // [rsp+48h] [rbp-268h]
  unsigned __int8 v116; // [rsp+57h] [rbp-259h]
  __int64 v118; // [rsp+68h] [rbp-248h] BYREF
  _QWORD v119[2]; // [rsp+70h] [rbp-240h] BYREF
  __int64 v120; // [rsp+80h] [rbp-230h]
  void *src; // [rsp+A0h] [rbp-210h] BYREF
  unsigned int v122; // [rsp+A8h] [rbp-208h]
  unsigned int v123; // [rsp+ACh] [rbp-204h]
  _BYTE v124[32]; // [rsp+B0h] [rbp-200h] BYREF
  __int64 v125; // [rsp+D0h] [rbp-1E0h] BYREF
  __int64 v126; // [rsp+D8h] [rbp-1D8h]
  _QWORD *v127; // [rsp+E0h] [rbp-1D0h] BYREF
  int v128; // [rsp+E8h] [rbp-1C8h]
  __int64 v129; // [rsp+100h] [rbp-1B0h] BYREF
  __int64 *v130; // [rsp+108h] [rbp-1A8h]
  void *s; // [rsp+110h] [rbp-1A0h]
  _BYTE v132[12]; // [rsp+118h] [rbp-198h]
  _BYTE v133[136]; // [rsp+128h] [rbp-188h] BYREF
  _BYTE *v134; // [rsp+1B0h] [rbp-100h] BYREF
  __int64 v135; // [rsp+1B8h] [rbp-F8h]
  _BYTE v136[240]; // [rsp+1C0h] [rbp-F0h] BYREF

  src = v124;
  v130 = (__int64 *)v133;
  s = v133;
  v115 = a2 + 40;
  v123 = 4;
  v129 = 0;
  *(_QWORD *)v132 = 16;
  *(_DWORD *)&v132[8] = 0;
  v116 = 0;
  while ( 2 )
  {
    v122 = 0;
    v15 = *(_QWORD *)(a2 + 48);
    if ( v15 == v115 )
    {
      v71 = (__int64 *)src;
      v72 = (__int64 *)src;
      goto LABEL_95;
    }
    do
    {
      if ( !v15 )
        BUG();
      if ( *(_BYTE *)(v15 - 8) != 77 )
        break;
      v16 = (char *)v130;
      v17 = v15 - 24;
      if ( s == v130 )
      {
        v18 = &v130[*(unsigned int *)&v132[4]];
        if ( v130 == v18 )
        {
          v102 = (char *)v130;
        }
        else
        {
          do
          {
            if ( v17 == *(_QWORD *)v16 )
              break;
            v16 += 8;
          }
          while ( v18 != (__int64 *)v16 );
          v102 = (char *)&v130[*(unsigned int *)&v132[4]];
        }
LABEL_88:
        while ( v102 != v16 )
        {
          if ( *(_QWORD *)v16 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_8;
          v16 += 8;
        }
        if ( v18 != (__int64 *)v16 )
          goto LABEL_9;
      }
      else
      {
        v18 = (__int64 *)((char *)s + 8 * *(unsigned int *)v132);
        v16 = (char *)sub_16CC9F0((__int64)&v129, v15 - 24);
        if ( v17 == *(_QWORD *)v16 )
        {
          if ( s == v130 )
            v102 = (char *)s + 8 * *(unsigned int *)&v132[4];
          else
            v102 = (char *)s + 8 * *(unsigned int *)v132;
          goto LABEL_88;
        }
        if ( s == v130 )
        {
          v16 = (char *)s + 8 * *(unsigned int *)&v132[4];
          v102 = v16;
          goto LABEL_88;
        }
        v16 = (char *)s + 8 * *(unsigned int *)v132;
LABEL_8:
        if ( v18 != (__int64 *)v16 )
          goto LABEL_9;
      }
      v70 = v122;
      if ( v122 >= v123 )
      {
        sub_16CD150((__int64)&src, v124, 0, 8, a13, a14);
        v70 = v122;
      }
      *((_QWORD *)src + v70) = v17;
      ++v122;
LABEL_9:
      v15 = *(_QWORD *)(v15 + 8);
    }
    while ( v15 != v115 );
    v72 = (__int64 *)src;
    v19 = 8LL * v122;
    v71 = (__int64 *)((char *)src + v19);
    if ( v19 )
    {
      v114 = a1;
      v20 = (__int64 *)src;
      v21 = (__int64 *)((char *)src + v19);
      v22 = v19 >> 3;
      do
      {
        v23 = (char *)sub_2207800(8 * v22, &unk_435FF63);
        v24 = v23;
        if ( v23 )
        {
          v25 = v20;
          a1 = v114;
          sub_1BCE0A0(v25, v21, v23, (void *)v22, (__int64 (__fastcall *)(__int64, __int64))sub_1BB9430);
          v26 = 8 * v22;
          goto LABEL_16;
        }
        v22 >>= 1;
      }
      while ( v22 );
      v71 = v21;
      v72 = v20;
      a1 = v114;
    }
LABEL_95:
    v24 = 0;
    sub_1BC67D0(v72, v71, (__int64 (__fastcall *)(__int64, __int64))sub_1BB9430);
    v26 = 0;
LABEL_16:
    j_j___libc_free_0(v24, v26);
    v29 = (__int64 ***)src;
    v30 = (__int64 ***)((char *)src + 8 * v122);
    if ( src == v30 )
      goto LABEL_38;
LABEL_17:
    v31 = v29;
LABEL_20:
    while ( 2 )
    {
      v34 = *v31;
      if ( **v31 == **v29 )
      {
        v32 = (__int64 *)s;
        v33 = v130;
        if ( s == v130 )
        {
          v35 = (__int64 ***)((char *)s + 8 * *(unsigned int *)&v132[4]);
          if ( s != v35 )
          {
            v36 = (__int64 ***)s;
            v37 = 0;
            while ( v34 != *v36 )
            {
              if ( *v36 == (__int64 **)-2LL )
                v37 = v36;
              if ( v35 == ++v36 )
              {
                if ( !v37 )
                  goto LABEL_33;
                ++v31;
                *v37 = v34;
                v32 = (__int64 *)s;
                --*(_DWORD *)&v132[8];
                v33 = v130;
                ++v129;
                if ( v30 != v31 )
                  goto LABEL_20;
                goto LABEL_30;
              }
            }
LABEL_19:
            if ( v30 == ++v31 )
            {
LABEL_30:
              v38 = v30 - v29;
              if ( (unsigned int)v38 <= 1 )
                goto LABEL_39;
              if ( (unsigned __int8)sub_1BDB410(
                                      a1,
                                      v29,
                                      (unsigned int)v38,
                                      (__int64)a3,
                                      0,
                                      (_DWORD)v38 == 2,
                                      a4,
                                      *(double *)a5.m128i_i64,
                                      *(double *)a6.m128i_i64,
                                      *(double *)a7.m128i_i64,
                                      v27,
                                      v28,
                                      a10,
                                      a11) )
                goto LABEL_32;
              goto LABEL_38;
            }
            continue;
          }
LABEL_33:
          if ( *(_DWORD *)&v132[4] < *(_DWORD *)v132 )
          {
            ++*(_DWORD *)&v132[4];
            *v35 = v34;
            v33 = v130;
            ++v129;
            v32 = (__int64 *)s;
            goto LABEL_19;
          }
        }
        sub_16CCBA0((__int64)&v129, (__int64)v34);
        v32 = (__int64 *)s;
        v33 = v130;
        goto LABEL_19;
      }
      break;
    }
    v39 = v31 - v29;
    if ( (unsigned int)v39 > 1
      && (unsigned __int8)sub_1BDB410(
                            a1,
                            v29,
                            (unsigned int)v39,
                            (__int64)a3,
                            0,
                            (_DWORD)v39 == 2,
                            a4,
                            *(double *)a5.m128i_i64,
                            *(double *)a6.m128i_i64,
                            *(double *)a7.m128i_i64,
                            v27,
                            v28,
                            a10,
                            a11) )
    {
LABEL_32:
      v116 = 1;
      continue;
    }
    break;
  }
  if ( v30 != v31 )
  {
    v29 = v31;
    goto LABEL_17;
  }
LABEL_38:
  v32 = (__int64 *)s;
  v33 = v130;
LABEL_39:
  ++v129;
  if ( v33 == v32 )
  {
LABEL_44:
    *(_QWORD *)&v132[4] = 0;
  }
  else
  {
    v40 = 4 * (*(_DWORD *)&v132[4] - *(_DWORD *)&v132[8]);
    if ( v40 < 0x20 )
      v40 = 32;
    if ( *(_DWORD *)v132 <= v40 )
    {
      memset(v32, -1, 8LL * *(unsigned int *)v132);
      goto LABEL_44;
    }
    sub_16CC920((__int64)&v129);
  }
  v125 = 0;
  v134 = v136;
  v135 = 0x800000000LL;
  v41 = (__int64 *)&v127;
  v126 = 1;
  do
    *v41++ = -8;
  while ( v41 != &v129 );
  v42 = a2;
  v43 = *(_QWORD *)(a2 + 48);
  if ( v43 != v115 )
  {
    v110 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    while ( 2 )
    {
      while ( 2 )
      {
        v64 = v43 - 24;
        if ( !v43 )
          v64 = 0;
        v65 = v130;
        if ( s != v130 )
          goto LABEL_49;
        v66 = &v130[*(unsigned int *)&v132[4]];
        if ( v130 != v66 )
        {
          v67 = 0;
          while ( v64 != *v65 )
          {
            if ( *v65 == -2 )
              v67 = v65;
            if ( v66 == ++v65 )
            {
              if ( !v67 )
                goto LABEL_136;
              *v67 = v64;
              --*(_DWORD *)&v132[8];
              v45 = *(unsigned __int8 *)(v64 + 16);
              ++v129;
              if ( (_BYTE)v45 != 78 )
                goto LABEL_51;
              goto LABEL_98;
            }
          }
          goto LABEL_78;
        }
LABEL_136:
        if ( *(_DWORD *)&v132[4] < *(_DWORD *)v132 )
        {
          ++*(_DWORD *)&v132[4];
          *v66 = v64;
          ++v129;
        }
        else
        {
LABEL_49:
          sub_16CCBA0((__int64)&v129, v64);
          if ( !v44 )
          {
LABEL_78:
            if ( !*(_QWORD *)(v64 + 8) )
            {
              if ( (v126 & 1) != 0 )
              {
                v68 = &v127;
                v69 = 3;
              }
              else
              {
                v68 = v127;
                if ( !v128 )
                  goto LABEL_68;
                v69 = v128 - 1;
              }
              v90 = 1;
              v91 = v69 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
              v92 = v68[v91];
              if ( v64 != v92 )
              {
                while ( v92 != -8 )
                {
                  LODWORD(v32) = v90 + 1;
                  v91 = v69 & (v90 + v91);
                  v92 = v68[v91];
                  if ( v64 == v92 )
                    goto LABEL_156;
                  ++v90;
                }
                goto LABEL_68;
              }
LABEL_156:
              v63 = sub_1BE01D0(a1, (__int64)&v134, v42, a3, (int)v32, a4, a5, a6, a7, v27, v28, a10, a11);
              if ( v63 )
              {
LABEL_67:
                v116 = v63;
                v43 = *(_QWORD *)(v42 + 48);
              }
            }
LABEL_68:
            v43 = *(_QWORD *)(v43 + 8);
            if ( v43 == v115 )
              goto LABEL_115;
            continue;
          }
        }
        break;
      }
      v45 = *(unsigned __int8 *)(v64 + 16);
      if ( (_BYTE)v45 == 78 )
      {
LABEL_98:
        v73 = *(_QWORD *)(v64 - 24);
        if ( *(_BYTE *)(v73 + 16) )
        {
          if ( *(_QWORD *)(v64 + 8) )
            goto LABEL_68;
          goto LABEL_140;
        }
        if ( (*(_BYTE *)(v73 + 33) & 0x20) != 0 )
        {
          if ( (unsigned int)(*(_DWORD *)(v73 + 36) - 35) <= 3 )
            goto LABEL_68;
          v74 = 3;
          if ( !*(_QWORD *)(v64 + 8) )
            goto LABEL_140;
        }
        else
        {
          v74 = 3;
          if ( !*(_QWORD *)(v64 + 8) )
            goto LABEL_140;
        }
      }
      else
      {
LABEL_51:
        if ( (_BYTE)v45 == 77 )
        {
          if ( (*(_DWORD *)(v64 + 20) & 0xFFFFFFF) != 2 )
            break;
          v46 = *(_QWORD *)(a1 + 32);
          v47 = *(_QWORD *)(a1 + 40);
          v112 = *(__int64 **)(a1 + 8);
          v48 = 24LL * *(unsigned int *)(v64 + 56) + 8;
          if ( (*(_BYTE *)(v64 + 23) & 0x40) != 0 )
          {
            v49 = *(__int64 **)(v64 - 8);
            v50 = (__int64 *)((char *)v49 + v48);
            if ( v42 != *v50 )
            {
LABEL_55:
              v51 = 0;
              if ( v42 != v50[1] || (v51 = v49[3]) == 0 )
              {
LABEL_56:
                v52 = *(_DWORD *)(v46 + 24);
                if ( !v52 )
                  goto LABEL_65;
                v53 = *(_QWORD *)(v46 + 8);
                v54 = v52 - 1;
                v55 = v54 & v110;
                v56 = (__int64 *)(v53 + 16LL * (v54 & v110));
                v57 = *v56;
                if ( v42 != *v56 )
                {
                  v103 = 1;
                  while ( v57 != -8 )
                  {
                    v104 = v103 + 1;
                    v105 = v54 & (unsigned int)(v55 + v103);
                    v55 = v105;
                    v56 = (__int64 *)(v53 + 16 * v105);
                    v57 = *v56;
                    if ( v42 == *v56 )
                      goto LABEL_58;
                    v103 = v104;
                  }
                  goto LABEL_65;
                }
LABEL_58:
                v58 = v56[1];
                v107 = v47;
                if ( v58 )
                {
                  v59 = sub_13FCB50(v58);
                  if ( v59 )
                  {
                    v60 = 24LL * *(unsigned int *)(v64 + 56) + 8;
                    if ( (*(_BYTE *)(v64 + 23) & 0x40) == 0 )
                    {
                      v61 = (__int64 *)(v64 - 24LL * (*(_DWORD *)(v64 + 20) & 0xFFFFFFF));
                      v62 = (__int64 *)((char *)v61 + v60);
                      if ( v59 == *v62 )
                        goto LABEL_132;
LABEL_62:
                      if ( v59 == v62[1] )
                      {
                        v51 = v61[3];
                        if ( !v51 )
                          goto LABEL_65;
                      }
                      else if ( !v51 )
                      {
                        goto LABEL_65;
                      }
                      if ( *(_BYTE *)(v51 + 16) > 0x17u )
                        goto LABEL_134;
                      goto LABEL_65;
                    }
                    v61 = *(__int64 **)(v64 - 8);
                    v62 = (__int64 *)((char *)v61 + v60);
                    if ( v59 != *v62 )
                      goto LABEL_62;
LABEL_132:
                    v51 = *v61;
                    if ( *v61 && *(_BYTE *)(v51 + 16) > 0x17u )
                    {
LABEL_134:
                      if ( !sub_15CC8F0(v107, *(_QWORD *)(v64 + 40), *(_QWORD *)(v51 + 40)) )
                        goto LABEL_65;
LABEL_66:
                      v63 = sub_1BE00E0(a1, v64, v51, v42, a3, v112, a4, a5, a6, a7, v27, v28, a10, a11);
                      if ( v63 )
                        goto LABEL_67;
                      goto LABEL_68;
                    }
                  }
                }
LABEL_65:
                v51 = 0;
                goto LABEL_66;
              }
LABEL_148:
              if ( *(_BYTE *)(v51 + 16) <= 0x17u )
                goto LABEL_56;
              v106 = *(_QWORD *)(a1 + 32);
              v108 = *(_QWORD *)(a1 + 40);
              v89 = sub_15CC8F0(v47, *(_QWORD *)(v64 + 40), *(_QWORD *)(v51 + 40));
              v47 = v108;
              v46 = v106;
              if ( !v89 )
                goto LABEL_56;
              goto LABEL_66;
            }
          }
          else
          {
            v49 = (__int64 *)(v64 - 48);
            v50 = (_QWORD *)(v64 - 48 + v48);
            if ( v42 != *v50 )
              goto LABEL_55;
          }
          v51 = *v49;
          if ( !*v49 )
            goto LABEL_56;
          goto LABEL_148;
        }
        if ( !*(_QWORD *)(v64 + 8) && (!*(_BYTE *)(*(_QWORD *)v64 + 8LL) || (_BYTE)v45 == 78 || (_BYTE)v45 == 29) )
        {
LABEL_140:
          v118 = v64;
          sub_1BCE550((__int64)v119, (__int64)&v125, &v118);
          v87 = byte_4FB9460;
          if ( byte_4FB9460 || *(_BYTE *)(v64 + 16) != 55 )
          {
            v93 = 3LL * (*(_DWORD *)(v64 + 20) & 0xFFFFFFF);
            if ( (*(_BYTE *)(v64 + 23) & 0x40) != 0 )
            {
              v95 = *(__int64 **)(v64 - 8);
              v94 = (__int64)&v95[v93];
            }
            else
            {
              v94 = v64;
              v95 = (__int64 *)(v64 - v93 * 8);
            }
            v87 = 0;
            if ( (__int64 *)v94 != v95 )
            {
              v113 = v43;
              v96 = v42;
              v97 = v64;
              v98 = v95;
              v99 = (__int64 *)v94;
              do
              {
                v100 = *v98;
                v109 = v96;
                v98 += 3;
                v101 = sub_1BE00E0(a1, 0, v100, v96, a3, *(__int64 **)(a1 + 8), a4, a5, a6, a7, v85, v86, a10, a11);
                v96 = v109;
                v87 |= v101;
              }
              while ( v99 != v98 );
              v64 = v97;
              v42 = v109;
              v43 = v113;
            }
          }
          v88 = sub_1BE01D0(a1, (__int64)&v134, v42, a3, v84, a4, a5, a6, a7, v85, v86, a10, a11) | v87;
          if ( v88 )
          {
            v116 = v88;
            v43 = *(_QWORD *)(v42 + 48);
            goto LABEL_68;
          }
          v45 = *(unsigned __int8 *)(v64 + 16);
        }
        v74 = (unsigned int)(v45 - 75);
        if ( (unsigned __int8)v74 > 0xCu )
          goto LABEL_68;
      }
      v75 = 4611;
      if ( !_bittest64(&v75, v74) )
        goto LABEL_68;
      v119[0] = 4;
      v119[1] = 0;
      v120 = v64;
      if ( v64 != -16 && v64 != -8 )
        sub_164C220((__int64)v119);
      v76 = v135;
      if ( (unsigned int)v135 >= HIDWORD(v135) )
      {
        sub_1BC1F40((__int64)&v134, 0);
        v76 = v135;
      }
      v77 = (unsigned __int64 *)&v134[24 * v76];
      if ( v77 )
      {
        *v77 = 4;
        v77[1] = 0;
        v78 = v120;
        v79 = v120 == -8;
        v77[2] = v120;
        if ( v78 != 0 && !v79 && v78 != -16 )
          sub_1649AC0(v77, v119[0] & 0xFFFFFFFFFFFFFFF8LL);
        v76 = v135;
      }
      LODWORD(v135) = v76 + 1;
      if ( v120 == -8 || v120 == 0 || v120 == -16 )
        goto LABEL_68;
      sub_1649B30(v119);
      v43 = *(_QWORD *)(v43 + 8);
      if ( v43 == v115 )
        break;
      continue;
    }
  }
LABEL_115:
  if ( (v126 & 1) == 0 )
    j___libc_free_0(v127);
  v80 = v134;
  v81 = &v134[24 * (unsigned int)v135];
  if ( v134 != (_BYTE *)v81 )
  {
    do
    {
      v82 = *(v81 - 1);
      v81 -= 3;
      if ( v82 != -8 && v82 != 0 && v82 != -16 )
        sub_1649B30(v81);
    }
    while ( v80 != v81 );
    v81 = v134;
  }
  if ( v81 != (_QWORD *)v136 )
    _libc_free((unsigned __int64)v81);
  if ( s != v130 )
    _libc_free((unsigned __int64)s);
  if ( src != v124 )
    _libc_free((unsigned __int64)src);
  return v116;
}
