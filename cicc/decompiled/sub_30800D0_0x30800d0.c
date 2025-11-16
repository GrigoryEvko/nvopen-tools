// Function: sub_30800D0
// Address: 0x30800d0
//
unsigned __int64 __fastcall sub_30800D0(__int64 a1, int a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  _DWORD *v13; // rdx
  int v14; // eax
  unsigned int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // ecx
  _DWORD *v18; // rdx
  int v19; // eax
  unsigned int v20; // esi
  __int64 v21; // rbx
  int v22; // r11d
  int *v23; // rdx
  __int64 v24; // r8
  unsigned int v25; // edi
  _DWORD *v26; // rax
  int v27; // ecx
  _QWORD *v28; // rax
  unsigned int v29; // ecx
  __int64 v30; // rdi
  __int64 v31; // rsi
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // r14
  unsigned __int64 result; // rax
  __int64 v39; // r15
  int v40; // r13d
  __int64 v41; // rsi
  unsigned int v42; // r8d
  int v43; // edi
  _DWORD *v44; // rax
  unsigned __int64 v45; // rsi
  unsigned int v46; // esi
  __int64 v47; // r8
  unsigned int v48; // r11d
  int *v49; // rdi
  int *v50; // r10
  int v51; // eax
  int v52; // esi
  int v53; // esi
  __int64 v54; // r11
  int *v55; // rdi
  int v56; // r8d
  unsigned int v57; // ecx
  int v58; // r9d
  int v59; // eax
  __int64 v60; // rsi
  __int64 v61; // rdi
  unsigned int v62; // ecx
  __int64 *v63; // rdx
  __int64 v64; // r9
  int v65; // eax
  int v66; // ecx
  int v67; // r9d
  int v68; // r11d
  _DWORD *v69; // r10
  int v70; // eax
  int v71; // edx
  int v72; // r11d
  _DWORD *v73; // r10
  int v74; // eax
  int v75; // edx
  int v76; // esi
  int v77; // esi
  __int64 v78; // r11
  unsigned int v79; // ecx
  int v80; // r9d
  int v81; // r10d
  int v82; // eax
  int v83; // esi
  __int64 v84; // r8
  __int64 v85; // rax
  int v86; // edi
  int v87; // r9d
  int *v88; // r10
  int v89; // eax
  int v90; // eax
  int v91; // r8d
  int *v92; // r9
  __int64 v93; // rdi
  __int64 v94; // r15
  int v95; // esi
  int v96; // eax
  int v97; // esi
  __int64 v98; // r8
  unsigned int v99; // eax
  int v100; // ecx
  int v101; // edi
  _DWORD *v102; // r9
  int v103; // eax
  int v104; // ecx
  __int64 v105; // rdi
  int v106; // esi
  _DWORD *v107; // r8
  unsigned int v108; // r15d
  int v109; // eax
  int v110; // eax
  int v111; // esi
  __int64 v112; // r9
  unsigned int v113; // eax
  int v114; // ecx
  int v115; // r8d
  _DWORD *v116; // rdi
  int v117; // eax
  int v118; // esi
  __int64 v119; // r8
  unsigned int v120; // r15d
  int v121; // ecx
  int v122; // eax
  int v123; // edx
  int v124; // r10d
  __int64 v125; // rdx
  int v126; // r8d
  int v127; // [rsp+18h] [rbp-48h]
  int v128; // [rsp+18h] [rbp-48h]
  int v129; // [rsp+18h] [rbp-48h]
  int v130[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v9 = *(_QWORD *)(a4 + 24);
  if ( v9 != *(_QWORD *)(a5 + 24) )
    goto LABEL_2;
  v60 = *(unsigned int *)(a1 + 136);
  v61 = *(_QWORD *)(a1 + 120);
  if ( (_DWORD)v60 )
  {
    v62 = (v60 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v63 = (__int64 *)(v61 + 16LL * v62);
    v64 = *v63;
    if ( v9 == *v63 )
      goto LABEL_39;
    v123 = 1;
    while ( v64 != -4096 )
    {
      v124 = v123 + 1;
      v125 = ((_DWORD)v60 - 1) & (v62 + v123);
      v62 = v125;
      v63 = (__int64 *)(v61 + 16 * v125);
      v64 = *v63;
      if ( v9 == *v63 )
        goto LABEL_39;
      v123 = v124;
    }
  }
  v63 = (__int64 *)(v61 + 16 * v60);
LABEL_39:
  if ( !(unsigned __int8)sub_307D8B0(a1, a2, v63[1]) )
    goto LABEL_4;
LABEL_2:
  v10 = *(_DWORD *)(a1 + 288);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 264);
    goto LABEL_114;
  }
  v11 = *(_QWORD *)(a1 + 272);
  v12 = (v10 - 1) & (37 * a3);
  v13 = (_DWORD *)(v11 + 4LL * v12);
  v14 = *v13;
  if ( *v13 == a3 )
    goto LABEL_4;
  v72 = 1;
  v73 = 0;
  while ( v14 != -1 )
  {
    if ( v14 != -2 || v73 )
      v13 = v73;
    v12 = (v10 - 1) & (v72 + v12);
    v14 = *(_DWORD *)(v11 + 4LL * v12);
    if ( v14 == a3 )
      goto LABEL_4;
    ++v72;
    v73 = v13;
    v13 = (_DWORD *)(v11 + 4LL * v12);
  }
  v74 = *(_DWORD *)(a1 + 280);
  if ( !v73 )
    v73 = v13;
  ++*(_QWORD *)(a1 + 264);
  v75 = v74 + 1;
  if ( 4 * (v74 + 1) >= 3 * v10 )
  {
LABEL_114:
    sub_A08C50(a1 + 264, 2 * v10);
    v110 = *(_DWORD *)(a1 + 288);
    if ( v110 )
    {
      v111 = v110 - 1;
      v112 = *(_QWORD *)(a1 + 272);
      v113 = (v110 - 1) & (37 * a3);
      v73 = (_DWORD *)(v112 + 4LL * v113);
      v114 = *v73;
      v75 = *(_DWORD *)(a1 + 280) + 1;
      if ( *v73 == a3 )
        goto LABEL_74;
      v115 = 1;
      v116 = 0;
      while ( v114 != -1 )
      {
        if ( v114 == -2 && !v116 )
          v116 = v73;
        v113 = v111 & (v115 + v113);
        v73 = (_DWORD *)(v112 + 4LL * v113);
        v114 = *v73;
        if ( *v73 == a3 )
          goto LABEL_74;
        ++v115;
      }
LABEL_118:
      if ( v116 )
        v73 = v116;
      goto LABEL_74;
    }
LABEL_186:
    ++*(_DWORD *)(a1 + 280);
    BUG();
  }
  if ( v10 - *(_DWORD *)(a1 + 284) - v75 <= v10 >> 3 )
  {
    sub_A08C50(a1 + 264, v10);
    v117 = *(_DWORD *)(a1 + 288);
    if ( v117 )
    {
      v118 = v117 - 1;
      v119 = *(_QWORD *)(a1 + 272);
      v116 = 0;
      v120 = (v117 - 1) & (37 * a3);
      v73 = (_DWORD *)(v119 + 4LL * v120);
      v121 = *v73;
      v75 = *(_DWORD *)(a1 + 280) + 1;
      v122 = 1;
      if ( *v73 == a3 )
        goto LABEL_74;
      while ( v121 != -1 )
      {
        if ( v121 == -2 && !v116 )
          v116 = v73;
        v120 = v118 & (v122 + v120);
        v73 = (_DWORD *)(v119 + 4LL * v120);
        v121 = *v73;
        if ( *v73 == a3 )
          goto LABEL_74;
        ++v122;
      }
      goto LABEL_118;
    }
    goto LABEL_186;
  }
LABEL_74:
  *(_DWORD *)(a1 + 280) = v75;
  if ( *v73 != -1 )
    --*(_DWORD *)(a1 + 284);
  *v73 = a3;
LABEL_4:
  v15 = *(_DWORD *)(a1 + 224);
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 200);
    goto LABEL_100;
  }
  v16 = *(_QWORD *)(a1 + 208);
  v17 = (v15 - 1) & (37 * a3);
  v18 = (_DWORD *)(v16 + 4LL * v17);
  v19 = *v18;
  if ( *v18 == a3 )
    goto LABEL_6;
  v68 = 1;
  v69 = 0;
  while ( v19 != -1 )
  {
    if ( v19 != -2 || v69 )
      v18 = v69;
    v17 = (v15 - 1) & (v68 + v17);
    v19 = *(_DWORD *)(v16 + 4LL * v17);
    if ( v19 == a3 )
      goto LABEL_6;
    ++v68;
    v69 = v18;
    v18 = (_DWORD *)(v16 + 4LL * v17);
  }
  v70 = *(_DWORD *)(a1 + 216);
  if ( !v69 )
    v69 = v18;
  ++*(_QWORD *)(a1 + 200);
  v71 = v70 + 1;
  if ( 4 * (v70 + 1) >= 3 * v15 )
  {
LABEL_100:
    sub_A08C50(a1 + 200, 2 * v15);
    v96 = *(_DWORD *)(a1 + 224);
    if ( v96 )
    {
      v97 = v96 - 1;
      v98 = *(_QWORD *)(a1 + 208);
      v99 = (v96 - 1) & (37 * a3);
      v69 = (_DWORD *)(v98 + 4LL * v99);
      v100 = *v69;
      v71 = *(_DWORD *)(a1 + 216) + 1;
      if ( *v69 != a3 )
      {
        v101 = 1;
        v102 = 0;
        while ( v100 != -1 )
        {
          if ( v100 == -2 && !v102 )
            v102 = v69;
          v99 = v97 & (v101 + v99);
          v69 = (_DWORD *)(v98 + 4LL * v99);
          v100 = *v69;
          if ( *v69 == a3 )
            goto LABEL_65;
          ++v101;
        }
        if ( v102 )
          v69 = v102;
      }
      goto LABEL_65;
    }
    goto LABEL_185;
  }
  if ( v15 - *(_DWORD *)(a1 + 220) - v71 <= v15 >> 3 )
  {
    sub_A08C50(a1 + 200, v15);
    v103 = *(_DWORD *)(a1 + 224);
    if ( v103 )
    {
      v104 = v103 - 1;
      v105 = *(_QWORD *)(a1 + 208);
      v106 = 1;
      v107 = 0;
      v108 = (v103 - 1) & (37 * a3);
      v69 = (_DWORD *)(v105 + 4LL * v108);
      v71 = *(_DWORD *)(a1 + 216) + 1;
      v109 = *v69;
      if ( *v69 != a3 )
      {
        while ( v109 != -1 )
        {
          if ( !v107 && v109 == -2 )
            v107 = v69;
          v108 = v104 & (v106 + v108);
          v69 = (_DWORD *)(v105 + 4LL * v108);
          v109 = *v69;
          if ( *v69 == a3 )
            goto LABEL_65;
          ++v106;
        }
        if ( v107 )
          v69 = v107;
      }
      goto LABEL_65;
    }
LABEL_185:
    ++*(_DWORD *)(a1 + 216);
    BUG();
  }
LABEL_65:
  *(_DWORD *)(a1 + 216) = v71;
  if ( *v69 != -1 )
    --*(_DWORD *)(a1 + 220);
  *v69 = a3;
LABEL_6:
  v20 = *(_DWORD *)(a1 + 320);
  v21 = *(_QWORD *)(a4 + 24);
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_86;
  }
  v22 = 1;
  v23 = 0;
  v24 = *(_QWORD *)(a1 + 304);
  v25 = (v20 - 1) & (37 * a2);
  v26 = (_DWORD *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( *v26 == a2 )
  {
LABEL_8:
    v28 = v26 + 2;
    goto LABEL_9;
  }
  while ( v27 != -1 )
  {
    if ( !v23 && v27 == -2 )
      v23 = v26;
    v25 = (v20 - 1) & (v22 + v25);
    v26 = (_DWORD *)(v24 + 16LL * v25);
    v27 = *v26;
    if ( *v26 == a2 )
      goto LABEL_8;
    ++v22;
  }
  if ( !v23 )
    v23 = v26;
  v65 = *(_DWORD *)(a1 + 312);
  ++*(_QWORD *)(a1 + 296);
  v66 = v65 + 1;
  if ( 4 * (v65 + 1) >= 3 * v20 )
  {
LABEL_86:
    sub_307FD10(a1 + 296, 2 * v20);
    v82 = *(_DWORD *)(a1 + 320);
    if ( v82 )
    {
      v83 = v82 - 1;
      v84 = *(_QWORD *)(a1 + 304);
      LODWORD(v85) = (v82 - 1) & (37 * a2);
      v66 = *(_DWORD *)(a1 + 312) + 1;
      v23 = (int *)(v84 + 16LL * (unsigned int)v85);
      v86 = *v23;
      if ( *v23 != a2 )
      {
        v87 = 1;
        v88 = 0;
        while ( v86 != -1 )
        {
          if ( v86 == -2 && !v88 )
            v88 = v23;
          v85 = v83 & (unsigned int)(v85 + v87);
          v23 = (int *)(v84 + 16 * v85);
          v86 = *v23;
          if ( *v23 == a2 )
            goto LABEL_51;
          ++v87;
        }
        if ( v88 )
          v23 = v88;
      }
      goto LABEL_51;
    }
    goto LABEL_188;
  }
  if ( v20 - *(_DWORD *)(a1 + 316) - v66 <= v20 >> 3 )
  {
    sub_307FD10(a1 + 296, v20);
    v89 = *(_DWORD *)(a1 + 320);
    if ( v89 )
    {
      v90 = v89 - 1;
      v91 = 1;
      v92 = 0;
      v93 = *(_QWORD *)(a1 + 304);
      LODWORD(v94) = v90 & (37 * a2);
      v66 = *(_DWORD *)(a1 + 312) + 1;
      v23 = (int *)(v93 + 16LL * (unsigned int)v94);
      v95 = *v23;
      if ( *v23 != a2 )
      {
        while ( v95 != -1 )
        {
          if ( !v92 && v95 == -2 )
            v92 = v23;
          v94 = v90 & (unsigned int)(v94 + v91);
          v23 = (int *)(v93 + 16 * v94);
          v95 = *v23;
          if ( *v23 == a2 )
            goto LABEL_51;
          ++v91;
        }
        if ( v92 )
          v23 = v92;
      }
      goto LABEL_51;
    }
LABEL_188:
    ++*(_DWORD *)(a1 + 312);
    BUG();
  }
LABEL_51:
  *(_DWORD *)(a1 + 312) = v66;
  if ( *v23 != -1 )
    --*(_DWORD *)(a1 + 316);
  *v23 = a2;
  v28 = v23 + 2;
  *((_QWORD *)v23 + 1) = 0;
LABEL_9:
  *v28 = v21;
  v29 = *(_DWORD *)(a1 + 136);
  v30 = *(_QWORD *)(a4 + 24);
  v31 = *(_QWORD *)(a1 + 120);
  if ( v29 )
  {
    v32 = (v29 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v33 = (__int64 *)(v31 + 16LL * v32);
    v34 = *v33;
    if ( v30 == *v33 )
      goto LABEL_11;
    v59 = 1;
    while ( v34 != -4096 )
    {
      v81 = v59 + 1;
      v32 = (v29 - 1) & (v59 + v32);
      v33 = (__int64 *)(v31 + 16LL * v32);
      v34 = *v33;
      if ( v30 == *v33 )
        goto LABEL_11;
      v59 = v81;
    }
  }
  v33 = (__int64 *)(v31 + 16LL * v29);
LABEL_11:
  v35 = v33[1];
  v36 = *(_QWORD *)(a4 + 32);
  v37 = v36 + 40LL * (*(_DWORD *)(a4 + 40) & 0xFFFFFF);
  result = (unsigned __int64)v130;
  if ( v37 != v36 )
  {
    v39 = v35;
    do
    {
      if ( !*(_BYTE *)v36 )
      {
        v40 = *(_DWORD *)(v36 + 8);
        if ( v40 < 0 )
        {
          result = *(unsigned int *)(a1 + 80);
          v41 = *(_QWORD *)(a1 + 64);
          v130[0] = *(_DWORD *)(v36 + 8);
          if ( (_DWORD)result )
          {
            result = (unsigned int)(result - 1);
            v42 = result & (37 * v40);
            v43 = *(_DWORD *)(v41 + 8LL * v42);
            if ( v40 != v43 )
            {
              v67 = 1;
              while ( v43 != -1 )
              {
                v42 = result & (v67 + v42);
                v43 = *(_DWORD *)(v41 + 8LL * v42);
                if ( v40 == v43 )
                  goto LABEL_18;
                ++v67;
              }
              goto LABEL_13;
            }
LABEL_18:
            v44 = sub_307C5F0(a1 + 56, v130);
            v45 = (unsigned int)*v44;
            result = *(_QWORD *)(*(_QWORD *)(v39 + 24) + 8LL * (*v44 >> 6));
            if ( _bittest64((const __int64 *)&result, v45) )
            {
              v46 = *(_DWORD *)(a1 + 256);
              if ( !v46 )
              {
                ++*(_QWORD *)(a1 + 232);
                goto LABEL_78;
              }
              v47 = *(_QWORD *)(a1 + 240);
              v48 = (v46 - 1) & (37 * v40);
              v49 = (int *)(v47 + 4LL * v48);
              result = (unsigned int)*v49;
              if ( v40 != (_DWORD)result )
              {
                v127 = 1;
                v50 = 0;
                while ( (_DWORD)result != -1 )
                {
                  if ( v50 || (_DWORD)result != -2 )
                    v49 = v50;
                  v48 = (v46 - 1) & (v127 + v48);
                  result = *(unsigned int *)(v47 + 4LL * v48);
                  if ( v40 == (_DWORD)result )
                    goto LABEL_13;
                  ++v127;
                  v50 = v49;
                  v49 = (int *)(v47 + 4LL * v48);
                }
                v51 = *(_DWORD *)(a1 + 248);
                if ( !v50 )
                  v50 = v49;
                ++*(_QWORD *)(a1 + 232);
                result = (unsigned int)(v51 + 1);
                if ( 4 * (int)result < 3 * v46 )
                {
                  if ( v46 - *(_DWORD *)(a1 + 252) - (unsigned int)result <= v46 >> 3 )
                  {
                    v128 = 37 * v40;
                    sub_A08C50(a1 + 232, v46);
                    v52 = *(_DWORD *)(a1 + 256);
                    if ( !v52 )
                      goto LABEL_187;
                    v53 = v52 - 1;
                    v54 = *(_QWORD *)(a1 + 240);
                    v55 = 0;
                    v56 = 1;
                    v57 = v53 & v128;
                    v50 = (int *)(v54 + 4LL * (v53 & (unsigned int)v128));
                    v58 = *v50;
                    result = (unsigned int)(*(_DWORD *)(a1 + 248) + 1);
                    if ( v40 != *v50 )
                    {
                      while ( v58 != -1 )
                      {
                        if ( !v55 && v58 == -2 )
                          v55 = v50;
                        v57 = v53 & (v57 + v56);
                        v50 = (int *)(v54 + 4LL * v57);
                        v58 = *v50;
                        if ( v40 == *v50 )
                          goto LABEL_80;
                        ++v56;
                      }
                      goto LABEL_30;
                    }
                  }
                  goto LABEL_80;
                }
LABEL_78:
                v129 = 37 * v40;
                sub_A08C50(a1 + 232, 2 * v46);
                v76 = *(_DWORD *)(a1 + 256);
                if ( !v76 )
                {
LABEL_187:
                  ++*(_DWORD *)(a1 + 248);
                  BUG();
                }
                v77 = v76 - 1;
                v78 = *(_QWORD *)(a1 + 240);
                v79 = v77 & v129;
                v50 = (int *)(v78 + 4LL * (v77 & (unsigned int)v129));
                v80 = *v50;
                result = (unsigned int)(*(_DWORD *)(a1 + 248) + 1);
                if ( v40 != *v50 )
                {
                  v126 = 1;
                  v55 = 0;
                  while ( v80 != -1 )
                  {
                    if ( v80 == -2 && !v55 )
                      v55 = v50;
                    v79 = v77 & (v79 + v126);
                    v50 = (int *)(v78 + 4LL * v79);
                    v80 = *v50;
                    if ( v40 == *v50 )
                      goto LABEL_80;
                    ++v126;
                  }
LABEL_30:
                  if ( v55 )
                    v50 = v55;
                }
LABEL_80:
                *(_DWORD *)(a1 + 248) = result;
                if ( *v50 != -1 )
                  --*(_DWORD *)(a1 + 252);
                *v50 = v40;
              }
            }
          }
        }
      }
LABEL_13:
      v36 += 40;
    }
    while ( v37 != v36 );
  }
  return result;
}
