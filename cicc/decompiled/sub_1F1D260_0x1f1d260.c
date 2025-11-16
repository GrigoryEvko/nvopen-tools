// Function: sub_1F1D260
// Address: 0x1f1d260
//
void __fastcall sub_1F1D260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  int v9; // r15d
  unsigned __int64 v10; // rcx
  unsigned int v11; // edx
  unsigned __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  _BYTE *v15; // rax
  int v16; // r13d
  __int64 v17; // rbx
  _BYTE *i; // rdx
  unsigned __int64 v19; // r13
  _BYTE *v20; // rax
  _BYTE *v21; // rdx
  __int64 *v22; // rbx
  unsigned int *v23; // r13
  __int64 v24; // r15
  unsigned int **v25; // rcx
  unsigned int **v26; // rax
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // r13
  __int64 v30; // rdi
  __int64 *v31; // rcx
  unsigned int v32; // ebx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdi
  __int64 *v37; // r15
  int v38; // r11d
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // r8
  unsigned int j; // eax
  unsigned int v42; // eax
  __int64 v43; // rcx
  __int64 v44; // rsi
  __int64 v45; // r13
  __int64 v46; // r13
  __int64 k; // rbx
  __int64 v48; // rcx
  unsigned __int64 *v49; // rax
  __int64 *v50; // r15
  unsigned int *v51; // r14
  __int64 v52; // rax
  int v53; // r9d
  int v54; // r8d
  __int64 v55; // rdx
  __int64 v56; // rcx
  unsigned __int64 v57; // rsi
  __int64 *v58; // rbx
  __int64 v59; // r15
  __int64 v60; // r14
  __int64 v61; // rdi
  int *v62; // r14
  __int64 v63; // rdx
  _QWORD *v64; // rax
  unsigned int v65; // eax
  int v66; // esi
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // rcx
  __int64 v70; // r8
  unsigned int v71; // ecx
  unsigned int v72; // r9d
  unsigned int v73; // edx
  __int64 *v74; // rdi
  __int64 v75; // r10
  __int64 *v76; // rax
  unsigned int v77; // edx
  __int64 *v78; // rdi
  __int64 v79; // r10
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rax
  _QWORD *v84; // rsi
  _QWORD *v85; // rax
  __int64 v86; // rcx
  __int64 v87; // rax
  __int64 v88; // rdx
  unsigned int *v89; // rax
  unsigned int v90; // edi
  int v91; // r11d
  unsigned int *v92; // r10
  int v93; // edx
  unsigned int *v94; // rsi
  int v95; // edi
  __int64 v96; // rax
  int v97; // edi
  int v98; // r11d
  int v99; // edi
  int v100; // edi
  int v101; // r11d
  __int64 v102; // [rsp+0h] [rbp-1C0h]
  __int64 v103; // [rsp+20h] [rbp-1A0h]
  __int64 v104; // [rsp+28h] [rbp-198h]
  unsigned __int64 v105; // [rsp+30h] [rbp-190h]
  __int64 v106; // [rsp+30h] [rbp-190h]
  __int64 v107; // [rsp+30h] [rbp-190h]
  __int64 v108; // [rsp+30h] [rbp-190h]
  __int64 *v109; // [rsp+38h] [rbp-188h]
  __int64 v110; // [rsp+38h] [rbp-188h]
  __int64 v111; // [rsp+38h] [rbp-188h]
  _QWORD *v112; // [rsp+38h] [rbp-188h]
  __int64 *v113; // [rsp+38h] [rbp-188h]
  __int64 v114; // [rsp+40h] [rbp-180h] BYREF
  unsigned __int64 v115; // [rsp+48h] [rbp-178h]
  __int64 v116; // [rsp+50h] [rbp-170h]
  __int64 v117; // [rsp+58h] [rbp-168h]
  _BYTE *v118; // [rsp+60h] [rbp-160h] BYREF
  __int64 v119; // [rsp+68h] [rbp-158h]
  _BYTE v120[64]; // [rsp+70h] [rbp-150h] BYREF
  _BYTE *v121; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v122; // [rsp+B8h] [rbp-108h]
  _BYTE v123[64]; // [rsp+C0h] [rbp-100h] BYREF
  _BYTE *v124; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v125; // [rsp+108h] [rbp-B8h]
  _BYTE v126[176]; // [rsp+110h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a1 + 72);
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_DWORD *)(**(_QWORD **)(v7 + 16) + 4LL * *(unsigned int *)(v7 + 64));
  v10 = *(unsigned int *)(v8 + 408);
  v11 = v9 & 0x7FFFFFFF;
  v12 = 8LL * (v9 & 0x7FFFFFFF);
  if ( (v9 & 0x7FFFFFFFu) >= (unsigned int)v10 || (v103 = *(_QWORD *)(*(_QWORD *)(v8 + 400) + 8LL * v11)) == 0 )
  {
    v32 = v11 + 1;
    if ( (unsigned int)v10 < v11 + 1 )
    {
      v83 = v32;
      if ( v32 < v10 )
      {
        *(_DWORD *)(v8 + 408) = v32;
      }
      else if ( v32 > v10 )
      {
        if ( v32 > (unsigned __int64)*(unsigned int *)(v8 + 412) )
        {
          v108 = 8LL * v11;
          sub_16CD150(v8 + 400, (const void *)(v8 + 416), v32, 8, 8 * v9, a6);
          v10 = *(unsigned int *)(v8 + 408);
          v12 = v108;
          v83 = v32;
        }
        v33 = *(_QWORD *)(v8 + 400);
        v84 = (_QWORD *)(v33 + 8 * v83);
        v85 = (_QWORD *)(v33 + 8 * v10);
        v86 = *(_QWORD *)(v8 + 416);
        if ( v84 != v85 )
        {
          do
            *v85++ = v86;
          while ( v84 != v85 );
          v33 = *(_QWORD *)(v8 + 400);
        }
        *(_DWORD *)(v8 + 408) = v32;
        goto LABEL_30;
      }
    }
    v33 = *(_QWORD *)(v8 + 400);
LABEL_30:
    *(_QWORD *)(v33 + v12) = sub_1DBA290(v9);
    v103 = *(_QWORD *)(*(_QWORD *)(v8 + 400) + 8LL * (v9 & 0x7FFFFFFF));
    sub_1DBB110((_QWORD *)v8, v103);
    v7 = *(_QWORD *)(a1 + 72);
  }
  v13 = *(_QWORD *)(v7 + 8);
  v125 = 0x800000000LL;
  v14 = *(unsigned int *)(v13 + 72);
  v104 = v13;
  v15 = v126;
  v124 = v126;
  v16 = v14;
  if ( (unsigned int)v14 > 8 )
  {
    sub_16CD150((__int64)&v124, v126, v14, 16, v12, a6);
    v15 = v124;
  }
  v17 = 16 * v14;
  LODWORD(v125) = v16;
  for ( i = &v15[v17]; i != v15; v15 += 16 )
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = 0;
      *((_QWORD *)v15 + 1) = 0;
    }
  }
  v119 = 0x800000000LL;
  v19 = *(unsigned int *)(v104 + 72);
  v20 = v120;
  v118 = v120;
  if ( (unsigned int)v19 > 8 )
  {
    sub_16CD150((__int64)&v118, v120, v19, 8, v12, a6);
    v20 = v118;
  }
  v21 = &v20[8 * v19];
  for ( LODWORD(v119) = v19; v21 != v20; v20 += 8 )
  {
    if ( v20 )
      *(_QWORD *)v20 = 0;
  }
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v22 = *(__int64 **)(v103 + 64);
  v117 = 0;
  v109 = &v22[*(unsigned int *)(v103 + 72)];
  if ( v22 != v109 )
  {
    while ( 1 )
    {
      v28 = *v22;
      v29 = *(_QWORD *)(*v22 + 8);
      if ( (v29 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_23;
      v30 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
      v31 = (__int64 *)sub_1DB3C70((__int64 *)v30, *(_QWORD *)(*v22 + 8));
      if ( v31 == (__int64 *)(*(_QWORD *)v30 + 24LL * *(unsigned int *)(v30 + 8))
        || (*(_DWORD *)((*v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v31 >> 1) & 3) > (*(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(v29 >> 1)
                                                                                              & 3) )
      {
        v23 = 0;
      }
      else
      {
        v23 = (unsigned int *)v31[2];
      }
      v24 = *(_QWORD *)(a1 + 72);
      v25 = *(unsigned int ***)(v24 + 168);
      v26 = *(unsigned int ***)(v24 + 160);
      if ( v25 == v26 )
      {
        v12 = (unsigned __int64)&v26[*(unsigned int *)(v24 + 180)];
        if ( v26 == (unsigned int **)v12 )
        {
          v10 = *(_QWORD *)(v24 + 160);
        }
        else
        {
          do
          {
            if ( v23 == *v26 )
              break;
            ++v26;
          }
          while ( (unsigned int **)v12 != v26 );
          v10 = v12;
        }
        goto LABEL_45;
      }
      v105 = (unsigned __int64)&v25[*(unsigned int *)(v24 + 176)];
      v26 = (unsigned int **)sub_16CC9F0(v24 + 152, (__int64)v23);
      v12 = v105;
      if ( v23 == *v26 )
        break;
      v27 = *(_QWORD *)(v24 + 168);
      if ( v27 == *(_QWORD *)(v24 + 160) )
      {
        v26 = (unsigned int **)(v27 + 8LL * *(unsigned int *)(v24 + 180));
        v10 = (unsigned __int64)v26;
LABEL_45:
        while ( (unsigned int **)v10 != v26 && (unsigned __int64)*v26 >= 0xFFFFFFFFFFFFFFFELL )
          ++v26;
        goto LABEL_22;
      }
      v10 = *(unsigned int *)(v24 + 176);
      v26 = (unsigned int **)(v27 + 8 * v10);
LABEL_22:
      if ( (unsigned int **)v12 != v26 )
        goto LABEL_23;
      v106 = *(_QWORD *)(v28 + 8);
      v34 = sub_1DA9310(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL), v106);
      v10 = v106;
      v35 = v34;
      v36 = *v23;
      v37 = (__int64 *)&v124[16 * v36];
      if ( *((_QWORD *)v23 + 1) == v106 )
      {
        *v37 = v34;
        v37[1] = v106;
        goto LABEL_23;
      }
      a6 = *(_DWORD *)(a1 + 424);
      if ( !a6 )
        goto LABEL_89;
      --a6;
      v38 = 1;
      v39 = (((unsigned int)(37 * v36) - 1LL - ((unsigned __int64)(unsigned int)(37 * v36) << 32)) >> 22)
          ^ ((unsigned int)(37 * v36) - 1LL - ((unsigned __int64)(unsigned int)(37 * v36) << 32));
      v40 = ((9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13)))) >> 15)
          ^ (9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13))));
      for ( j = a6 & (((v40 - 1 - (v40 << 27)) >> 31) ^ (v40 - 1 - ((_DWORD)v40 << 27))); ; j = a6 & v42 )
      {
        v12 = *(_QWORD *)(a1 + 408) + 16LL * j;
        if ( !*(_DWORD *)v12 )
          break;
        if ( *(_DWORD *)v12 == -1 && *(_DWORD *)(v12 + 4) == -1 )
          goto LABEL_89;
LABEL_37:
        v42 = v38 + j;
        ++v38;
      }
      if ( (_DWORD)v36 != *(_DWORD *)(v12 + 4) )
        goto LABEL_37;
      if ( (*(_QWORD *)(v12 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_23;
LABEL_89:
      if ( !*v37 )
      {
        v87 = *(_QWORD *)(v28 + 8);
        *v37 = v35;
        v37[1] = v87;
        goto LABEL_23;
      }
      if ( *v37 == v35 )
      {
        if ( (v37[1] & 0xFFFFFFFFFFFFFFF8LL) == 0
          || (v12 = v106 & 0xFFFFFFFFFFFFFFF8LL,
              (*(_DWORD *)((v106 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v106 >> 1) & 3) < (*(_DWORD *)((v37[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                 | (unsigned int)(v37[1] >> 1)
                                                                                                 & 3)) )
        {
          v37[1] = v106;
        }
        goto LABEL_23;
      }
      v107 = *v37;
      v102 = *(_QWORD *)(a1 + 40);
      sub_1E06620(v102);
      v68 = *(_QWORD *)(*(_QWORD *)(v107 + 56) + 328LL);
      if ( v107 == v68 )
        goto LABEL_105;
      if ( v35 == v68 )
        goto LABEL_141;
      v69 = *(_QWORD *)(v102 + 1312);
      v70 = *(_QWORD *)(v69 + 32);
      v71 = *(_DWORD *)(v69 + 48);
      if ( !v71 )
        goto LABEL_143;
      v72 = v71 - 1;
      v73 = (v71 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
      v74 = (__int64 *)(v70 + 16LL * v73);
      v75 = *v74;
      if ( v107 == *v74 )
      {
LABEL_95:
        v76 = (__int64 *)(v70 + 16LL * v71);
        if ( v76 != v74 )
        {
          v68 = v74[1];
          goto LABEL_97;
        }
      }
      else
      {
        v99 = 1;
        while ( v75 != -8 )
        {
          v101 = v99 + 1;
          v73 = v72 & (v99 + v73);
          v74 = (__int64 *)(v70 + 16LL * v73);
          v75 = *v74;
          if ( v107 == *v74 )
            goto LABEL_95;
          v99 = v101;
        }
        v76 = (__int64 *)(v70 + 16LL * v71);
      }
      v68 = 0;
LABEL_97:
      v77 = v72 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v78 = (__int64 *)(v70 + 16LL * v77);
      v79 = *v78;
      if ( v35 != *v78 )
      {
        v97 = 1;
        while ( v79 != -8 )
        {
          v98 = v97 + 1;
          v77 = v72 & (v97 + v77);
          v78 = (__int64 *)(v70 + 16LL * v77);
          v79 = *v78;
          if ( v35 == *v78 )
            goto LABEL_98;
          v97 = v98;
        }
LABEL_143:
        v68 = 0;
        goto LABEL_105;
      }
LABEL_98:
      if ( v78 == v76 )
        goto LABEL_143;
      v80 = v78[1];
      if ( !v68 || !v80 )
        goto LABEL_143;
      while ( v68 != v80 )
      {
        if ( *(_DWORD *)(v68 + 16) < *(_DWORD *)(v80 + 16) )
        {
          v81 = v68;
          v68 = v80;
          v80 = v81;
        }
        v68 = *(_QWORD *)(v68 + 8);
        if ( !v68 )
          goto LABEL_105;
      }
      v68 = *(_QWORD *)v68;
LABEL_105:
      if ( v35 == v68 )
      {
LABEL_141:
        v96 = *(_QWORD *)(v28 + 8);
        *v37 = v35;
        v37[1] = v96;
        goto LABEL_108;
      }
      if ( *v37 != v68 )
      {
        *v37 = v68;
        v37[1] = 0;
      }
LABEL_108:
      v82 = sub_1DDC3C0(*(_QWORD *)(a1 + 64), v35);
      sub_16AF570(&v118[8 * *v23], v82);
LABEL_23:
      if ( v109 == ++v22 )
        goto LABEL_50;
    }
    v43 = *(_QWORD *)(v24 + 168);
    if ( v43 == *(_QWORD *)(v24 + 160) )
      v44 = *(unsigned int *)(v24 + 180);
    else
      v44 = *(unsigned int *)(v24 + 176);
    v10 = v43 + 8 * v44;
    goto LABEL_45;
  }
LABEL_50:
  v45 = *(unsigned int *)(v104 + 72);
  if ( (_DWORD)v45 )
  {
    v46 = 8 * v45;
    for ( k = 0; v46 != k; k += 8 )
    {
      v50 = (__int64 *)&v124[2 * k];
      v12 = *v50;
      v111 = *v50;
      if ( *v50 && (v50[1] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        v51 = *(unsigned int **)(*(_QWORD *)(v104 + 64) + k);
        v52 = sub_1DA9310(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL), *((_QWORD *)v51 + 1));
        v54 = v111;
        v55 = v52;
        if ( v111 != v52 )
          v55 = sub_1F13550((_QWORD *)a1, v111, v52);
        *v50 = v55;
        if ( *(_DWORD *)(a1 + 84) != 2 )
          goto LABEL_60;
        v112 = &v118[8 * *v51];
        if ( *v112 >= (unsigned __int64)sub_1DDC3C0(*(_QWORD *)(a1 + 64), v55) )
        {
          v55 = *v50;
LABEL_60:
          v56 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL) + 392LL)
                          + 16LL * *(unsigned int *)(v55 + 48)
                          + 8);
          v57 = v56 & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((v56 >> 1) & 3) != 0 )
            v48 = v57 | (2LL * (int)(((v56 >> 1) & 3) - 1));
          else
            v48 = *(_QWORD *)v57 & 0xFFFFFFFFFFFFFFF8LL | 6;
          v110 = v48;
          v49 = (unsigned __int64 *)sub_1F14200(
                                      (_QWORD *)(*(_QWORD *)a1 + 48LL),
                                      *(_QWORD *)(*(_QWORD *)a1 + 40LL),
                                      v55,
                                      v48,
                                      v54,
                                      v53);
          v50[1] = *(_QWORD *)(sub_1F1AD70((_QWORD *)a1, 0, (int *)v51, v110, *v50, v49) + 8);
          continue;
        }
        if ( !(_DWORD)v117 )
        {
          ++v114;
          goto LABEL_145;
        }
        v10 = *v51;
        a6 = v117 - 1;
        v12 = v115;
        LODWORD(v88) = (v117 - 1) & (37 * v10);
        v89 = (unsigned int *)(v115 + 4LL * (unsigned int)v88);
        v90 = *v89;
        if ( (_DWORD)v10 != *v89 )
        {
          v91 = 1;
          v92 = 0;
          while ( v90 != -1 )
          {
            if ( v90 == -2 && !v92 )
              v92 = v89;
            v88 = a6 & ((_DWORD)v88 + v91);
            v89 = (unsigned int *)(v115 + 4 * v88);
            v90 = *v89;
            if ( (_DWORD)v10 == *v89 )
              goto LABEL_54;
            ++v91;
          }
          if ( v92 )
            v89 = v92;
          ++v114;
          v93 = v116 + 1;
          if ( 4 * ((int)v116 + 1) < (unsigned int)(3 * v117) )
          {
            v10 = (unsigned int)(v117 - HIDWORD(v116) - v93);
            if ( (unsigned int)v10 <= (unsigned int)v117 >> 3 )
            {
              sub_136B240((__int64)&v114, v117);
              if ( !(_DWORD)v117 )
                goto LABEL_181;
              a6 = *v51;
              v94 = 0;
              v95 = 1;
              v93 = v116 + 1;
              v10 = ((_DWORD)v117 - 1) & (37 * *v51);
              v89 = (unsigned int *)(v115 + 4 * v10);
              v12 = *v89;
              if ( (_DWORD)v12 != *v51 )
              {
                while ( (_DWORD)v12 != -1 )
                {
                  if ( !v94 && (_DWORD)v12 == -2 )
                    v94 = v89;
                  v10 = ((_DWORD)v117 - 1) & (unsigned int)(v10 + v95);
                  v89 = (unsigned int *)(v115 + 4 * v10);
                  v12 = *v89;
                  if ( a6 == (_DWORD)v12 )
                    goto LABEL_147;
                  ++v95;
                }
                goto LABEL_135;
              }
            }
            goto LABEL_147;
          }
LABEL_145:
          sub_136B240((__int64)&v114, 2 * v117);
          if ( !(_DWORD)v117 )
          {
LABEL_181:
            LODWORD(v116) = v116 + 1;
            BUG();
          }
          v12 = *v51;
          v93 = v116 + 1;
          v10 = ((_DWORD)v117 - 1) & (unsigned int)(37 * v12);
          v89 = (unsigned int *)(v115 + 4 * v10);
          a6 = *v89;
          if ( (_DWORD)v12 != *v89 )
          {
            v100 = 1;
            v94 = 0;
            while ( a6 != -1 )
            {
              if ( !v94 && a6 == -2 )
                v94 = v89;
              v10 = ((_DWORD)v117 - 1) & (unsigned int)(v10 + v100);
              v89 = (unsigned int *)(v115 + 4 * v10);
              a6 = *v89;
              if ( (_DWORD)v12 == *v89 )
                goto LABEL_147;
              ++v100;
            }
LABEL_135:
            if ( v94 )
              v89 = v94;
          }
LABEL_147:
          LODWORD(v116) = v93;
          if ( *v89 != -1 )
            --HIDWORD(v116);
          *v89 = *v51;
        }
      }
LABEL_54:
      ;
    }
  }
  v121 = v123;
  v122 = 0x800000000LL;
  v58 = *(__int64 **)(v103 + 64);
  if ( v58 != &v58[*(unsigned int *)(v103 + 72)] )
  {
    v113 = &v58[*(unsigned int *)(v103 + 72)];
    while ( 1 )
    {
      v59 = *v58;
      v60 = *(_QWORD *)(*v58 + 8);
      if ( (v60 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_73;
      v61 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
      v10 = sub_1DB3C70((__int64 *)v61, *(_QWORD *)(*v58 + 8));
      if ( v10 == *(_QWORD *)v61 + 24LL * *(unsigned int *)(v61 + 8)
        || (*(_DWORD *)((*(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v10 >> 1) & 3) > (*(_DWORD *)((v60 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v60 >> 1) & 3) )
      {
        BUG();
      }
      v62 = *(int **)(v10 + 16);
      v63 = (unsigned int)*v62;
      v64 = &v124[16 * v63];
      if ( !*v64 || *(_QWORD *)(v59 + 8) == v64[1] )
        goto LABEL_73;
      if ( !(_DWORD)v117 )
        goto LABEL_84;
      v10 = (unsigned int)(v117 - 1);
      v65 = v10 & (37 * v63);
      v66 = *(_DWORD *)(v115 + 4LL * v65);
      if ( (_DWORD)v63 != v66 )
        break;
LABEL_73:
      if ( v113 == ++v58 )
        goto LABEL_74;
    }
    v12 = 1;
    while ( v66 != -1 )
    {
      a6 = v12 + 1;
      v65 = v10 & (v12 + v65);
      v66 = *(_DWORD *)(v115 + 4LL * v65);
      if ( (_DWORD)v63 == v66 )
        goto LABEL_73;
      v12 = a6;
    }
LABEL_84:
    v67 = (unsigned int)v122;
    if ( (unsigned int)v122 >= HIDWORD(v122) )
    {
      sub_16CD150((__int64)&v121, v123, 0, 8, v12, a6);
      v67 = (unsigned int)v122;
    }
    *(_QWORD *)&v121[8 * v67] = v59;
    LODWORD(v122) = v122 + 1;
    sub_1F1B3E0(a1, 0, v62);
    goto LABEL_73;
  }
LABEL_74:
  if ( *(_DWORD *)(a1 + 84) == 2 && (_DWORD)v116 )
    sub_1F1C430((_QWORD *)a1, (__int64)&v114, (__int64)&v121, v10, v12, a6);
  sub_1F1BDF0(a1, (__int64)&v121);
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  j___libc_free_0(v115);
  if ( v118 != v120 )
    _libc_free((unsigned __int64)v118);
  if ( v124 != v126 )
    _libc_free((unsigned __int64)v124);
}
