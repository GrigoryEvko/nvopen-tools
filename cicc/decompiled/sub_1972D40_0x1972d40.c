// Function: sub_1972D40
// Address: 0x1972d40
//
__int64 __fastcall sub_1972D40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *i; // rdx
  __int64 v14; // rbx
  char *v15; // rax
  char *v16; // r12
  __int128 v17; // rdi
  __int64 v18; // r8
  __int64 *v19; // r15
  unsigned int v20; // r14d
  int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rbx
  char *v25; // rax
  _QWORD *v26; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // rcx
  __int64 v33; // rax
  int v34; // esi
  __int64 v35; // r9
  int v36; // esi
  unsigned int v37; // edi
  __int64 *v38; // rdx
  __int64 v39; // r11
  _QWORD *v40; // rdx
  unsigned int v41; // edi
  __int64 *v42; // rcx
  __int64 v43; // r11
  _QWORD *j; // rax
  __int64 *v45; // r13
  char *v46; // rax
  char *v47; // r9
  _QWORD *v48; // rax
  __int64 *v49; // r15
  _QWORD *v50; // r12
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  _QWORD *v53; // rax
  _QWORD *v54; // rdx
  __int64 v55; // rsi
  _QWORD *v56; // rax
  _QWORD *v57; // r9
  __int64 v58; // rax
  _QWORD *v59; // rax
  int v60; // r9d
  void *v61; // rdi
  unsigned int v62; // eax
  __int64 v63; // rdx
  unsigned int v64; // eax
  __int64 *v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rsi
  char *v69; // rsi
  char *v70; // rcx
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rsi
  _QWORD *v74; // rdi
  unsigned int v75; // r9d
  _QWORD *v76; // rsi
  _QWORD *v77; // rsi
  char *v78; // rdx
  _QWORD *v79; // rax
  int v80; // ecx
  int v81; // edx
  _QWORD *v82; // rdx
  char *v84; // [rsp+10h] [rbp-2C0h]
  __int64 v87; // [rsp+28h] [rbp-2A8h]
  __int64 v88; // [rsp+30h] [rbp-2A0h]
  char *v89; // [rsp+38h] [rbp-298h]
  __int64 v92; // [rsp+58h] [rbp-278h]
  __int64 *v93; // [rsp+68h] [rbp-268h]
  _QWORD *v95; // [rsp+80h] [rbp-250h]
  __int64 v96; // [rsp+88h] [rbp-248h]
  __int64 v97; // [rsp+90h] [rbp-240h]
  char *v98; // [rsp+98h] [rbp-238h]
  __int64 v99; // [rsp+98h] [rbp-238h]
  _QWORD *v100; // [rsp+A0h] [rbp-230h]
  int v101; // [rsp+A8h] [rbp-228h]
  int v102; // [rsp+ACh] [rbp-224h]
  __m128i v103; // [rsp+B0h] [rbp-220h] BYREF
  __int64 v104; // [rsp+C0h] [rbp-210h]
  __int64 v105; // [rsp+C8h] [rbp-208h]
  __int64 v106; // [rsp+D0h] [rbp-200h]
  _QWORD v107[2]; // [rsp+E0h] [rbp-1F0h] BYREF
  _QWORD *v108; // [rsp+F0h] [rbp-1E0h]
  __int64 v109; // [rsp+F8h] [rbp-1D8h]
  unsigned int v110; // [rsp+100h] [rbp-1D0h]
  void *src; // [rsp+108h] [rbp-1C8h]
  char *v112; // [rsp+110h] [rbp-1C0h]
  char *v113; // [rsp+118h] [rbp-1B8h]
  __int64 v114; // [rsp+120h] [rbp-1B0h] BYREF
  char *v115; // [rsp+128h] [rbp-1A8h]
  void *s; // [rsp+130h] [rbp-1A0h]
  _BYTE v117[12]; // [rsp+138h] [rbp-198h]
  _BYTE v118[40]; // [rsp+148h] [rbp-188h] BYREF
  _BYTE *v119; // [rsp+170h] [rbp-160h] BYREF
  __int64 v120; // [rsp+178h] [rbp-158h]
  _BYTE v121[64]; // [rsp+180h] [rbp-150h] BYREF
  __int64 v122; // [rsp+1C0h] [rbp-110h] BYREF
  _BYTE *v123; // [rsp+1C8h] [rbp-108h]
  _BYTE *v124; // [rsp+1D0h] [rbp-100h]
  __int64 v125; // [rsp+1D8h] [rbp-F8h]
  int v126; // [rsp+1E0h] [rbp-F0h]
  _BYTE v127[72]; // [rsp+1E8h] [rbp-E8h] BYREF
  __int64 v128; // [rsp+230h] [rbp-A0h] BYREF
  _BYTE *v129; // [rsp+238h] [rbp-98h]
  _BYTE *v130; // [rsp+240h] [rbp-90h]
  __int64 v131; // [rsp+248h] [rbp-88h]
  int v132; // [rsp+250h] [rbp-80h]
  _BYTE v133[120]; // [rsp+258h] [rbp-78h] BYREF

  v5 = sub_157EB90(**(_QWORD **)(a1 + 32));
  v107[0] = a1;
  v92 = sub_1632FA0(v5);
  v123 = v127;
  v124 = v127;
  v129 = v133;
  v130 = v133;
  v115 = v118;
  s = v118;
  v119 = v121;
  v120 = 0x800000000LL;
  v6 = *(_QWORD *)(v107[0] + 40LL) - *(_QWORD *)(v107[0] + 32LL);
  v122 = 0;
  v125 = 8;
  v7 = (unsigned int)(v6 >> 3);
  v126 = 0;
  v132 = 0;
  v128 = 0;
  v131 = 8;
  v114 = 0;
  *(_QWORD *)v117 = 4;
  *(_DWORD *)&v117[8] = 0;
  v107[1] = 0;
  v8 = ((((((((v7 | (v7 >> 1)) >> 2) | v7 | (v7 >> 1)) >> 4) | ((v7 | (v7 >> 1)) >> 2) | v7 | (v7 >> 1)) >> 8)
       | ((((v7 | (v7 >> 1)) >> 2) | v7 | (v7 >> 1)) >> 4)
       | ((v7 | (v7 >> 1)) >> 2)
       | v7
       | (v7 >> 1)) >> 16)
     | ((((((v7 | (v7 >> 1)) >> 2) | v7 | (v7 >> 1)) >> 4) | ((v7 | (v7 >> 1)) >> 2) | v7 | (v7 >> 1)) >> 8)
     | ((((v7 | (v7 >> 1)) >> 2) | v7 | (v7 >> 1)) >> 4)
     | ((v7 | (v7 >> 1)) >> 2)
     | v7
     | (v7 >> 1);
  if ( (_DWORD)v8 == -1 )
  {
    v108 = 0;
    v109 = 0;
    v110 = 0;
  }
  else
  {
    v9 = (4 * ((int)v8 + 1) / 3u + 1) | ((unsigned __int64)(4 * ((int)v8 + 1) / 3u + 1) >> 1);
    v10 = (((v9 >> 2) | v9) >> 4) | (v9 >> 2) | v9;
    v11 = ((((v10 >> 8) | v10) >> 16) | (v10 >> 8) | v10) + 1;
    v110 = v11;
    v12 = (_QWORD *)sub_22077B0(16 * v11);
    v109 = 0;
    v108 = v12;
    for ( i = &v12[2 * v110]; i != v12; v12 += 2 )
    {
      if ( v12 )
        *v12 = -8;
    }
    v7 = (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3);
  }
  src = 0;
  v112 = 0;
  v113 = 0;
  if ( v7 )
  {
    v14 = 8 * v7;
    v15 = (char *)sub_22077B0(v14);
    v16 = v15;
    if ( v112 - (_BYTE *)src > 0 )
    {
      memmove(v15, src, v112 - (_BYTE *)src);
      j_j___libc_free_0(src, v113 - (_BYTE *)src);
    }
    src = v16;
    v112 = v16;
    v113 = &v16[v14];
  }
  *((_QWORD *)&v17 + 1) = a3;
  *(_QWORD *)&v17 = v107;
  sub_13FF3D0(v17);
  v93 = &v128;
  v19 = &v122;
  v20 = 0;
  while ( 2 )
  {
    v89 = v112;
    v84 = (char *)src;
    if ( v112 == src )
      goto LABEL_67;
    v21 = v20;
    do
    {
      v22 = *((_QWORD *)v89 - 1);
      v23 = *(_QWORD *)(v22 + 48);
      v97 = v22 + 40;
      if ( v22 + 40 == v23 )
        goto LABEL_65;
      do
      {
        while ( 1 )
        {
          if ( !v23 )
            BUG();
          v24 = v23 - 24;
          if ( *(_BYTE *)(v23 - 8) == 77 )
          {
            v25 = v115;
            if ( s != v115 )
              goto LABEL_19;
            v69 = &v115[8 * *(unsigned int *)&v117[4]];
            if ( v115 == v69 )
            {
LABEL_149:
              if ( *(_DWORD *)&v117[4] >= *(_DWORD *)v117 )
              {
LABEL_19:
                sub_16CCBA0((__int64)&v114, v23 - 24);
                goto LABEL_20;
              }
              ++*(_DWORD *)&v117[4];
              *(_QWORD *)v69 = v24;
              ++v114;
            }
            else
            {
              v70 = 0;
              while ( v24 != *(_QWORD *)v25 )
              {
                if ( *(_QWORD *)v25 == -2 )
                  v70 = v25;
                v25 += 8;
                if ( v69 == v25 )
                {
                  if ( !v70 )
                    goto LABEL_149;
                  *(_QWORD *)v70 = v24;
                  --*(_DWORD *)&v117[8];
                  ++v114;
                  if ( *(_QWORD *)(v23 - 16) )
                    goto LABEL_21;
                  goto LABEL_122;
                }
              }
            }
          }
LABEL_20:
          if ( !*(_QWORD *)(v23 - 16) )
          {
LABEL_122:
            if ( (unsigned __int8)sub_1AE9990(v23 - 24, a5) )
              goto LABEL_123;
            goto LABEL_64;
          }
LABEL_21:
          v102 = *((_DWORD *)v19 + 7);
          v101 = *((_DWORD *)v19 + 8);
          if ( v102 != v101 )
          {
            v26 = (_QWORD *)v19[2];
            v27 = (_QWORD *)v19[1];
            if ( v26 == v27 )
            {
              v28 = &v27[v102];
              if ( v27 == v28 )
              {
                v82 = (_QWORD *)v19[1];
              }
              else
              {
                do
                {
                  if ( v24 == *v27 )
                    break;
                  ++v27;
                }
                while ( v28 != v27 );
                v82 = v28;
              }
            }
            else
            {
              v100 = &v26[*((unsigned int *)v19 + 6)];
              v27 = sub_16CC9F0((__int64)v19, v23 - 24);
              v28 = v100;
              if ( v24 == *v27 )
              {
                v72 = v19[2];
                if ( v72 == v19[1] )
                  v73 = *((unsigned int *)v19 + 7);
                else
                  v73 = *((unsigned int *)v19 + 6);
                v82 = (_QWORD *)(v72 + 8 * v73);
              }
              else
              {
                v29 = v19[2];
                if ( v29 != v19[1] )
                {
                  v27 = (_QWORD *)(v29 + 8LL * *((unsigned int *)v19 + 6));
                  goto LABEL_26;
                }
                v27 = (_QWORD *)(v29 + 8LL * *((unsigned int *)v19 + 7));
                v82 = v27;
              }
            }
            while ( v82 != v27 && *v27 >= 0xFFFFFFFFFFFFFFFELL )
              ++v27;
LABEL_26:
            if ( v28 == v27 )
              goto LABEL_64;
          }
          v106 = v23 - 24;
          v103.m128i_i64[0] = v92;
          v103.m128i_i64[1] = a5;
          v104 = a2;
          v105 = a4;
          v30 = sub_13E3350(v23 - 24, &v103, 0, 1, v18);
          v31 = v30;
          if ( !v30 )
            goto LABEL_64;
          if ( *(_BYTE *)(v30 + 16) <= 0x17u )
            break;
          v32 = *(_QWORD *)(v30 + 40);
          v33 = *(_QWORD *)(v23 + 16);
          if ( v32 == v33 )
            break;
          v34 = *(_DWORD *)(a3 + 24);
          if ( !v34 )
            break;
          v35 = *(_QWORD *)(a3 + 8);
          v36 = v34 - 1;
          v37 = v36 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
          v38 = (__int64 *)(v35 + 16LL * v37);
          v39 = *v38;
          if ( v32 != *v38 )
          {
            v81 = 1;
            while ( v39 != -8 )
            {
              v18 = (unsigned int)(v81 + 1);
              v37 = v36 & (v37 + v81);
              v38 = (__int64 *)(v35 + 16LL * v37);
              v39 = *v38;
              if ( v32 == *v38 )
                goto LABEL_32;
              v81 = v18;
            }
            break;
          }
LABEL_32:
          v40 = (_QWORD *)v38[1];
          if ( !v40 )
            break;
          v41 = v36 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
          v42 = (__int64 *)(v35 + 16LL * v41);
          v43 = *v42;
          if ( *v42 == v33 )
          {
LABEL_34:
            for ( j = (_QWORD *)v42[1]; v40 != j; j = (_QWORD *)*j )
            {
              if ( !j )
                goto LABEL_64;
            }
            break;
          }
          v80 = 1;
          while ( v43 != -8 )
          {
            v18 = (unsigned int)(v80 + 1);
            v41 = v36 & (v41 + v80);
            v42 = (__int64 *)(v35 + 16LL * v41);
            v43 = *v42;
            if ( v33 == *v42 )
              goto LABEL_34;
            v80 = v18;
          }
LABEL_64:
          v23 = *(_QWORD *)(v23 + 8);
          if ( v97 == v23 )
            goto LABEL_65;
        }
        v45 = *(__int64 **)(v23 - 16);
        if ( !v45 )
          goto LABEL_62;
        v88 = v23 - 24;
        v87 = v23;
        v96 = (__int64)v19;
        do
        {
          while ( 1 )
          {
            v49 = v45;
            v45 = (__int64 *)v45[1];
            v50 = sub_1648700((__int64)v49);
            if ( *v49 )
            {
              v51 = v49[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v51 = v45;
              if ( v45 )
                v45[2] = v45[2] & 3 | v51;
            }
            *v49 = v31;
            v52 = *(_QWORD *)(v31 + 8);
            v49[1] = v52;
            if ( v52 )
              *(_QWORD *)(v52 + 16) = (unsigned __int64)(v49 + 1) | *(_QWORD *)(v52 + 16) & 3LL;
            v49[2] = (v31 + 8) | v49[2] & 3;
            *(_QWORD *)(v31 + 8) = v49;
            if ( *((_BYTE *)v50 + 16) != 77 )
              break;
            v46 = v115;
            if ( s == v115 )
            {
              v47 = &v115[8 * *(unsigned int *)&v117[4]];
              if ( v115 == v47 )
              {
                v78 = v115;
              }
              else
              {
                do
                {
                  if ( v50 == *(_QWORD **)v46 )
                    break;
                  v46 += 8;
                }
                while ( v47 != v46 );
                v78 = &v115[8 * *(unsigned int *)&v117[4]];
              }
            }
            else
            {
              v98 = (char *)s + 8 * *(unsigned int *)v117;
              v46 = (char *)sub_16CC9F0((__int64)&v114, (__int64)v50);
              v47 = v98;
              if ( v50 == *(_QWORD **)v46 )
              {
                if ( s == v115 )
                  v78 = (char *)s + 8 * *(unsigned int *)&v117[4];
                else
                  v78 = (char *)s + 8 * *(unsigned int *)v117;
              }
              else
              {
                if ( s != v115 )
                {
                  v46 = (char *)s + 8 * *(unsigned int *)v117;
                  goto LABEL_43;
                }
                v46 = (char *)s + 8 * *(unsigned int *)&v117[4];
                v78 = v46;
              }
            }
            while ( v78 != v46 && *(_QWORD *)v46 >= 0xFFFFFFFFFFFFFFFELL )
              v46 += 8;
LABEL_43:
            if ( v47 == v46 )
              break;
            v48 = (_QWORD *)v93[1];
            if ( (_QWORD *)v93[2] == v48 )
            {
              v74 = &v48[*((unsigned int *)v93 + 7)];
              v75 = *((_DWORD *)v93 + 7);
              if ( v48 != v74 )
              {
                v77 = 0;
                while ( v50 != (_QWORD *)*v48 )
                {
                  if ( *v48 == -2 )
                    v77 = v48;
                  if ( v74 == ++v48 )
                  {
                    if ( !v77 )
                      goto LABEL_161;
                    *v77 = v50;
                    --*((_DWORD *)v93 + 8);
                    ++*v93;
                    goto LABEL_46;
                  }
                }
                goto LABEL_46;
              }
LABEL_161:
              v79 = v93;
              if ( v75 < *((_DWORD *)v93 + 6) )
                goto LABEL_152;
            }
            sub_16CCBA0((__int64)v93, (__int64)v50);
LABEL_46:
            if ( !v45 )
              goto LABEL_61;
          }
          if ( v102 == v101 )
            goto LABEL_46;
          v53 = *(_QWORD **)(a1 + 72);
          v54 = *(_QWORD **)(a1 + 64);
          v55 = v50[5];
          if ( v53 == v54 )
          {
            v66 = &v54[*(unsigned int *)(a1 + 84)];
            if ( v54 == v66 )
            {
              v57 = *(_QWORD **)(a1 + 64);
            }
            else
            {
              do
              {
                if ( v55 == *v54 )
                  break;
                ++v54;
              }
              while ( v66 != v54 );
              v57 = v66;
            }
          }
          else
          {
            v99 = v50[5];
            v95 = &v53[*(unsigned int *)(a1 + 80)];
            v56 = sub_16CC9F0(a1 + 56, v55);
            v57 = v95;
            v54 = v56;
            if ( v99 == *v56 )
            {
              v67 = *(_QWORD *)(a1 + 72);
              if ( v67 == *(_QWORD *)(a1 + 64) )
                v68 = *(unsigned int *)(a1 + 84);
              else
                v68 = *(unsigned int *)(a1 + 80);
              v66 = (_QWORD *)(v67 + 8 * v68);
            }
            else
            {
              v58 = *(_QWORD *)(a1 + 72);
              if ( v58 != *(_QWORD *)(a1 + 64) )
              {
                v54 = (_QWORD *)(v58 + 8LL * *(unsigned int *)(a1 + 80));
                goto LABEL_58;
              }
              v66 = (_QWORD *)(v58 + 8LL * *(unsigned int *)(a1 + 84));
              v54 = v66;
            }
          }
          while ( v66 != v54 && *v54 >= 0xFFFFFFFFFFFFFFFELL )
            ++v54;
LABEL_58:
          if ( v57 == v54 )
            goto LABEL_46;
          v59 = *(_QWORD **)(v96 + 8);
          if ( *(_QWORD **)(v96 + 16) != v59 )
            goto LABEL_60;
          v74 = &v59[*(unsigned int *)(v96 + 28)];
          v75 = *(_DWORD *)(v96 + 28);
          if ( v59 != v74 )
          {
            v76 = 0;
            while ( v50 != (_QWORD *)*v59 )
            {
              if ( *v59 == -2 )
                v76 = v59;
              if ( v74 == ++v59 )
              {
                if ( !v76 )
                  goto LABEL_151;
                *v76 = v50;
                --*(_DWORD *)(v96 + 32);
                ++*(_QWORD *)v96;
                goto LABEL_46;
              }
            }
            goto LABEL_46;
          }
LABEL_151:
          v79 = (_QWORD *)v96;
          if ( v75 < *(_DWORD *)(v96 + 24) )
          {
LABEL_152:
            *((_DWORD *)v79 + 7) = v75 + 1;
            *v74 = v50;
            ++*v79;
            goto LABEL_46;
          }
LABEL_60:
          sub_16CCBA0(v96, (__int64)v50);
        }
        while ( v45 );
LABEL_61:
        v24 = v88;
        v23 = v87;
        v19 = (__int64 *)v96;
LABEL_62:
        v21 = sub_1AE9990(v24, a5);
        if ( !(_BYTE)v21 )
        {
          v21 = 1;
          goto LABEL_64;
        }
LABEL_123:
        v71 = (unsigned int)v120;
        if ( (unsigned int)v120 >= HIDWORD(v120) )
        {
          sub_16CD150((__int64)&v119, v121, 0, 8, v18, v60);
          v71 = (unsigned int)v120;
        }
        *(_QWORD *)&v119[8 * v71] = v24;
        LODWORD(v120) = v120 + 1;
        v23 = *(_QWORD *)(v23 + 8);
      }
      while ( v97 != v23 );
LABEL_65:
      v89 -= 8;
    }
    while ( v84 != v89 );
    v20 = v21;
LABEL_67:
    if ( (_DWORD)v120 )
    {
      v20 = 1;
      sub_1AEB210(&v119, a5);
      if ( *((_DWORD *)v93 + 7) == *((_DWORD *)v93 + 8) )
        goto LABEL_164;
LABEL_69:
      ++*v19;
      v61 = (void *)v19[2];
      if ( v61 != (void *)v19[1] )
      {
        v62 = 4 * (*((_DWORD *)v19 + 7) - *((_DWORD *)v19 + 8));
        v63 = *((unsigned int *)v19 + 6);
        if ( v62 < 0x20 )
          v62 = 32;
        if ( (unsigned int)v63 > v62 )
        {
          sub_16CC920((__int64)v19);
LABEL_75:
          ++v114;
          if ( s != v115 )
          {
            v64 = 4 * (*(_DWORD *)&v117[4] - *(_DWORD *)&v117[8]);
            if ( v64 < 0x20 )
              v64 = 32;
            if ( *(_DWORD *)v117 > v64 )
            {
              sub_16CC920((__int64)&v114);
              goto LABEL_81;
            }
            memset(s, -1, 8LL * *(unsigned int *)v117);
          }
          *(_QWORD *)&v117[4] = 0;
LABEL_81:
          LODWORD(v120) = 0;
          v65 = v19;
          v19 = v93;
          v93 = v65;
          continue;
        }
        memset(v61, -1, 8 * v63);
      }
      *(__int64 *)((char *)v19 + 28) = 0;
      goto LABEL_75;
    }
    break;
  }
  if ( *((_DWORD *)v93 + 7) != *((_DWORD *)v93 + 8) )
    goto LABEL_69;
LABEL_164:
  if ( src )
    j_j___libc_free_0(src, v113 - (_BYTE *)src);
  j___libc_free_0(v108);
  if ( v119 != v121 )
    _libc_free((unsigned __int64)v119);
  if ( s != v115 )
    _libc_free((unsigned __int64)s);
  if ( v130 != v129 )
    _libc_free((unsigned __int64)v130);
  if ( v124 != v123 )
    _libc_free((unsigned __int64)v124);
  return v20;
}
