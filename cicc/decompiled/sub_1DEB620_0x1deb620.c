// Function: sub_1DEB620
// Address: 0x1deb620
//
unsigned __int64 __fastcall sub_1DEB620(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 v5; // r15
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 *v9; // rbx
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // rbx
  char v18; // r13
  __int64 v19; // r11
  unsigned __int8 v20; // al
  char v21; // al
  unsigned int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // ebx
  unsigned int v25; // edi
  __int64 *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r14
  __int64 *v29; // r13
  __int64 v30; // rax
  __int64 *v31; // rbx
  __int64 v32; // rdx
  __int64 *v33; // r8
  __int64 *v34; // r9
  __int64 *v35; // r13
  __int64 v36; // r11
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rbx
  unsigned int v44; // esi
  int v45; // esi
  int v46; // esi
  unsigned int v47; // ecx
  int v48; // edx
  __int64 v49; // rdi
  unsigned __int64 result; // rax
  __int64 v51; // rbx
  _QWORD *v52; // rcx
  __int64 v53; // rdi
  int v54; // esi
  int v55; // r9d
  __int64 v56; // r8
  unsigned int v57; // esi
  __int64 v58; // r9
  unsigned int v59; // r8d
  __int64 v60; // rdi
  int v61; // r10d
  int v62; // edi
  int v63; // ecx
  int v64; // ecx
  __int64 v65; // rdi
  unsigned int v66; // r15d
  int v67; // r10d
  __int64 v68; // rsi
  __int64 *v69; // rdx
  int v70; // eax
  int v71; // eax
  int v72; // r9d
  int v73; // r9d
  __int64 v74; // rdi
  __int64 *v75; // r11
  __int64 v76; // rcx
  __int64 v77; // rsi
  int v78; // r10d
  int v79; // r8d
  int v80; // r8d
  __int64 v81; // rsi
  int v82; // r9d
  __int64 v83; // r13
  __int64 v84; // rcx
  int v85; // r10d
  __int64 *v86; // rdx
  int v87; // eax
  int v88; // eax
  char v89; // al
  __int64 v90; // rax
  int v91; // esi
  int v92; // esi
  __int64 v93; // r8
  unsigned int v94; // ecx
  __int64 v95; // rdi
  int v96; // r10d
  __int64 *v97; // r9
  int v98; // ecx
  int v99; // ecx
  int v100; // r9d
  __int64 *v101; // r8
  unsigned int v102; // ebx
  __int64 v103; // rdi
  __int64 v104; // rsi
  int v105; // r15d
  __int64 *v106; // r13
  int v107; // [rsp+14h] [rbp-7Ch]
  __int64 *v108; // [rsp+18h] [rbp-78h]
  __int64 v109; // [rsp+20h] [rbp-70h]
  const void *v110; // [rsp+28h] [rbp-68h]
  char v112; // [rsp+38h] [rbp-58h]
  __int64 v113; // [rsp+38h] [rbp-58h]
  __int64 v114; // [rsp+38h] [rbp-58h]
  __int64 v115; // [rsp+38h] [rbp-58h]
  __int64 *v117; // [rsp+48h] [rbp-48h]
  __int64 *v118; // [rsp+48h] [rbp-48h]
  __int64 v119; // [rsp+48h] [rbp-48h]
  __int64 v120; // [rsp+48h] [rbp-48h]
  __int64 v121; // [rsp+48h] [rbp-48h]
  __int64 v122; // [rsp+48h] [rbp-48h]
  _QWORD *v123; // [rsp+50h] [rbp-40h] BYREF
  __int64 v124[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = a3;
  v5 = a1;
  v6 = *(__int64 **)a3;
  v123 = *(_QWORD **)(*(_QWORD *)(a1 + 552) + 328LL);
  v7 = *(unsigned int *)(a3 + 8);
  v8 = (__int64)&v6[v7];
  if ( v6 != (__int64 *)v8 )
  {
    v117 = &v6[v7];
    v9 = v6;
    do
    {
      v12 = *v9++;
      sub_1DE6B80(a1, (__int64 **)a3, v12, a2, a4);
    }
    while ( v117 != v9 );
    v4 = a3;
    v8 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  }
  v13 = v4;
  v109 = a1 + 888;
  v110 = (const void *)(v4 + 16);
  v14 = *(_QWORD *)(v8 - 8);
  while ( 1 )
  {
    v15 = sub_1DEA040(v5, v14, (void ***)v13, a4);
    v112 = v16;
    v17 = v15;
    v18 = v16;
    v19 = v15;
    if ( !byte_4FC5020 || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v5 + 552) + 8LL) + 640LL) & 1) != 0 )
    {
      if ( v15 )
        goto LABEL_14;
    }
    else if ( v15 )
    {
      v20 = sub_1F34100(v15);
      v19 = v17;
      if ( (unsigned int)((__int64)(*(_QWORD *)(v17 + 96) - *(_QWORD *)(v17 + 88)) >> 3) != 1 )
      {
        v21 = sub_1F34340(v5 + 616, v20, v17);
        v19 = v17;
        v112 = v18 | v21;
      }
      goto LABEL_11;
    }
    v19 = sub_1DE61F0(v5, v13, v5 + 232);
    if ( !v19 )
    {
      v19 = sub_1DE61F0(v5, v13, v5 + 376);
      if ( !v19 )
        break;
    }
LABEL_11:
    if ( byte_4FC5020
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v5 + 552) + 8LL) + 640LL) & 1) == 0
      && v112
      && (v120 = v19, v89 = sub_1DE5B90(v5, v19, v14, v13, a4, (__int64)&v123, v124), v19 = v120, v89) )
    {
      if ( !LOBYTE(v124[0]) )
        goto LABEL_34;
      do
      {
        v90 = *(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8);
        if ( *(_QWORD *)v13 == v90 - 8 )
        {
          v14 = *(_QWORD *)(v90 - 8);
          goto LABEL_109;
        }
        sub_1DE5B90(v5, *(_QWORD *)(v90 - 8), *(_QWORD *)(v90 - 16), v13, a4, (__int64)&v123, v124);
      }
      while ( LOBYTE(v124[0]) );
      v14 = *(_QWORD *)(*(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8) - 8);
LABEL_109:
      sub_1DE6B80(v5, (__int64 **)v13, v14, a2, a4);
    }
    else
    {
LABEL_14:
      v22 = *(_DWORD *)(v5 + 912);
      if ( v22 )
      {
        v23 = *(_QWORD *)(v5 + 896);
        v24 = ((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4);
        v25 = (v22 - 1) & v24;
        v26 = (__int64 *)(v23 + 16LL * v25);
        v27 = *v26;
        if ( *v26 == v19 )
        {
          v28 = v26[1];
          goto LABEL_17;
        }
        v85 = 1;
        v86 = 0;
        while ( v27 != -8 )
        {
          if ( v27 != -16 || v86 )
            v26 = v86;
          v25 = (v22 - 1) & (v85 + v25);
          v106 = (__int64 *)(v23 + 16LL * v25);
          v27 = *v106;
          if ( *v106 == v19 )
          {
            v28 = v106[1];
            goto LABEL_17;
          }
          ++v85;
          v86 = v26;
          v26 = (__int64 *)(v23 + 16LL * v25);
        }
        if ( !v86 )
          v86 = v26;
        v87 = *(_DWORD *)(v5 + 904);
        ++*(_QWORD *)(v5 + 888);
        v88 = v87 + 1;
        if ( 4 * v88 < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(v5 + 908) - v88 <= v22 >> 3 )
          {
            v122 = v19;
            sub_1DE4DF0(v109, v22);
            v98 = *(_DWORD *)(v5 + 912);
            if ( !v98 )
            {
LABEL_174:
              ++*(_DWORD *)(v5 + 904);
              BUG();
            }
            v99 = v98 - 1;
            v19 = v122;
            v100 = 1;
            v101 = 0;
            v102 = v99 & v24;
            v103 = *(_QWORD *)(v5 + 896);
            v88 = *(_DWORD *)(v5 + 904) + 1;
            v86 = (__int64 *)(v103 + 16LL * v102);
            v104 = *v86;
            if ( *v86 != v122 )
            {
              while ( v104 != -8 )
              {
                if ( !v101 && v104 == -16 )
                  v101 = v86;
                v102 = v99 & (v100 + v102);
                v86 = (__int64 *)(v103 + 16LL * v102);
                v104 = *v86;
                if ( *v86 == v122 )
                  goto LABEL_100;
                ++v100;
              }
              if ( v101 )
                v86 = v101;
            }
          }
          goto LABEL_100;
        }
      }
      else
      {
        ++*(_QWORD *)(v5 + 888);
      }
      v121 = v19;
      sub_1DE4DF0(v109, 2 * v22);
      v91 = *(_DWORD *)(v5 + 912);
      if ( !v91 )
        goto LABEL_174;
      v19 = v121;
      v92 = v91 - 1;
      v93 = *(_QWORD *)(v5 + 896);
      v94 = v92 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
      v88 = *(_DWORD *)(v5 + 904) + 1;
      v86 = (__int64 *)(v93 + 16LL * v94);
      v95 = *v86;
      if ( *v86 != v121 )
      {
        v96 = 1;
        v97 = 0;
        while ( v95 != -8 )
        {
          if ( !v97 && v95 == -16 )
            v97 = v86;
          v94 = v92 & (v96 + v94);
          v86 = (__int64 *)(v93 + 16LL * v94);
          v95 = *v86;
          if ( *v86 == v121 )
            goto LABEL_100;
          ++v96;
        }
        if ( v97 )
          v86 = v97;
      }
LABEL_100:
      *(_DWORD *)(v5 + 904) = v88;
      if ( *v86 != -8 )
        --*(_DWORD *)(v5 + 908);
      *v86 = v19;
      v28 = 0;
      v86[1] = 0;
LABEL_17:
      v29 = *(__int64 **)v28;
      v30 = *(unsigned int *)(v28 + 8);
      *(_DWORD *)(v28 + 56) = 0;
      if ( v29 != &v29[v30] )
      {
        v118 = &v29[v30];
        v31 = v29;
        do
        {
          v32 = *v31++;
          sub_1DE6B80(v5, (__int64 **)v28, v32, a2, a4);
        }
        while ( v118 != v31 );
        v35 = *(__int64 **)v28;
        v36 = *(_QWORD *)v28 + 8LL * *(unsigned int *)(v28 + 8);
        if ( *(_QWORD *)v28 != v36 )
        {
          v119 = v5;
          while ( 1 )
          {
            v41 = *v35;
            v42 = *(unsigned int *)(v13 + 8);
            if ( (unsigned int)v42 >= *(_DWORD *)(v13 + 12) )
            {
              v114 = v36;
              sub_16CD150(v13, v110, 0, 8, (int)v33, (int)v34);
              v42 = *(unsigned int *)(v13 + 8);
              v36 = v114;
            }
            *(_QWORD *)(*(_QWORD *)v13 + 8 * v42) = v41;
            v43 = *(_QWORD *)(v13 + 48);
            ++*(_DWORD *)(v13 + 8);
            v44 = *(_DWORD *)(v43 + 24);
            if ( !v44 )
              break;
            LODWORD(v33) = v44 - 1;
            v37 = *(_QWORD *)(v43 + 8);
            v38 = (v44 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
            v39 = (__int64 *)(v37 + 16LL * v38);
            v40 = *v39;
            if ( v41 == *v39 )
            {
LABEL_23:
              ++v35;
              v39[1] = v13;
              if ( (__int64 *)v36 == v35 )
                goto LABEL_33;
            }
            else
            {
              v61 = 1;
              v34 = 0;
              while ( v40 != -8 )
              {
                if ( !v34 && v40 == -16 )
                  v34 = v39;
                v38 = (unsigned int)v33 & (v61 + v38);
                v39 = (__int64 *)(v37 + 16LL * v38);
                v40 = *v39;
                if ( v41 == *v39 )
                  goto LABEL_23;
                ++v61;
              }
              v62 = *(_DWORD *)(v43 + 16);
              if ( v34 )
                v39 = v34;
              ++*(_QWORD *)v43;
              v48 = v62 + 1;
              if ( 4 * (v62 + 1) < 3 * v44 )
              {
                if ( v44 - *(_DWORD *)(v43 + 20) - v48 <= v44 >> 3 )
                {
                  v115 = v36;
                  sub_1DE4DF0(v43, v44);
                  v63 = *(_DWORD *)(v43 + 24);
                  if ( !v63 )
                  {
LABEL_173:
                    ++*(_DWORD *)(v43 + 16);
                    BUG();
                  }
                  v64 = v63 - 1;
                  v65 = *(_QWORD *)(v43 + 8);
                  v33 = 0;
                  v66 = v64 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                  v36 = v115;
                  v67 = 1;
                  v48 = *(_DWORD *)(v43 + 16) + 1;
                  v39 = (__int64 *)(v65 + 16LL * v66);
                  v68 = *v39;
                  if ( v41 != *v39 )
                  {
                    while ( v68 != -8 )
                    {
                      if ( !v33 && v68 == -16 )
                        v33 = v39;
                      LODWORD(v34) = v67 + 1;
                      v66 = v64 & (v67 + v66);
                      v39 = (__int64 *)(v65 + 16LL * v66);
                      v68 = *v39;
                      if ( v41 == *v39 )
                        goto LABEL_30;
                      ++v67;
                    }
                    if ( v33 )
                      v39 = v33;
                  }
                }
                goto LABEL_30;
              }
LABEL_28:
              v113 = v36;
              sub_1DE4DF0(v43, 2 * v44);
              v45 = *(_DWORD *)(v43 + 24);
              if ( !v45 )
                goto LABEL_173;
              v46 = v45 - 1;
              v33 = *(__int64 **)(v43 + 8);
              v36 = v113;
              v47 = v46 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v48 = *(_DWORD *)(v43 + 16) + 1;
              v39 = &v33[2 * v47];
              v49 = *v39;
              if ( v41 != *v39 )
              {
                v105 = 1;
                v34 = 0;
                while ( v49 != -8 )
                {
                  if ( !v34 && v49 == -16 )
                    v34 = v39;
                  v47 = v46 & (v105 + v47);
                  v39 = &v33[2 * v47];
                  v49 = *v39;
                  if ( v41 == *v39 )
                    goto LABEL_30;
                  ++v105;
                }
                if ( v34 )
                  v39 = v34;
              }
LABEL_30:
              *(_DWORD *)(v43 + 16) = v48;
              if ( *v39 != -8 )
                --*(_DWORD *)(v43 + 20);
              ++v35;
              v39[1] = 0;
              *v39 = v41;
              v39[1] = v13;
              if ( (__int64 *)v36 == v35 )
              {
LABEL_33:
                v5 = v119;
                goto LABEL_34;
              }
            }
          }
          ++*(_QWORD *)v43;
          goto LABEL_28;
        }
      }
LABEL_34:
      v14 = *(_QWORD *)(*(_QWORD *)v13 + 8LL * *(unsigned int *)(v13 + 8) - 8);
    }
  }
  result = *(_QWORD *)(v5 + 552);
  v51 = (__int64)v123;
  v52 = (_QWORD *)(result + 320);
  if ( v123 == (_QWORD *)(result + 320) )
    return result;
  while ( 2 )
  {
    while ( a4 )
    {
      if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
      {
        v53 = a4 + 16;
        v54 = 15;
      }
      else
      {
        result = *(unsigned int *)(a4 + 24);
        v53 = *(_QWORD *)(a4 + 16);
        if ( !(_DWORD)result )
          goto LABEL_45;
        v54 = result - 1;
      }
      v55 = 1;
      result = v54 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v56 = *(_QWORD *)(v53 + 8 * result);
      if ( v56 == v51 )
        break;
      while ( v56 != -8 )
      {
        result = v54 & (unsigned int)(v55 + result);
        v56 = *(_QWORD *)(v53 + 8LL * (unsigned int)result);
        if ( v56 == v51 )
          goto LABEL_42;
        ++v55;
      }
      v51 = *(_QWORD *)(v51 + 8);
      if ( v52 == (_QWORD *)v51 )
        return result;
    }
LABEL_42:
    v57 = *(_DWORD *)(v5 + 912);
    if ( !v57 )
    {
      ++*(_QWORD *)(v5 + 888);
LABEL_79:
      sub_1DE4DF0(v109, 2 * v57);
      v72 = *(_DWORD *)(v5 + 912);
      if ( v72 )
      {
        v73 = v72 - 1;
        v74 = *(_QWORD *)(v5 + 896);
        v75 = 0;
        LODWORD(v76) = v73 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
        v71 = *(_DWORD *)(v5 + 904) + 1;
        v69 = (__int64 *)(v74 + 16LL * (unsigned int)v76);
        v77 = *v69;
        if ( *v69 == v51 )
          goto LABEL_73;
        v78 = 1;
        while ( v77 != -8 )
        {
          if ( v77 == -16 && !v75 )
            v75 = v69;
          v76 = v73 & (unsigned int)(v76 + v78);
          v69 = (__int64 *)(v74 + 16 * v76);
          v77 = *v69;
          if ( *v69 == v51 )
            goto LABEL_73;
          ++v78;
        }
LABEL_83:
        if ( v75 )
          v69 = v75;
        goto LABEL_73;
      }
LABEL_175:
      ++*(_DWORD *)(v5 + 904);
      BUG();
    }
    v58 = *(_QWORD *)(v5 + 896);
    v59 = (v57 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    result = v58 + 16LL * v59;
    v60 = *(_QWORD *)result;
    if ( v51 == *(_QWORD *)result )
    {
LABEL_44:
      if ( v13 != *(_QWORD *)(result + 8) )
        goto LABEL_76;
LABEL_45:
      v51 = *(_QWORD *)(v51 + 8);
      if ( v52 == (_QWORD *)v51 )
        return result;
      continue;
    }
    break;
  }
  v108 = 0;
  v107 = 1;
  while ( v60 != -8 )
  {
    if ( v60 == -16 )
    {
      if ( v108 )
        result = (unsigned __int64)v108;
      v108 = (__int64 *)result;
    }
    v59 = (v57 - 1) & (v107 + v59);
    result = v58 + 16LL * v59;
    v60 = *(_QWORD *)result;
    if ( *(_QWORD *)result == v51 )
      goto LABEL_44;
    ++v107;
  }
  v69 = v108;
  if ( !v108 )
    v69 = (__int64 *)result;
  v70 = *(_DWORD *)(v5 + 904);
  ++*(_QWORD *)(v5 + 888);
  v71 = v70 + 1;
  if ( 4 * v71 >= 3 * v57 )
    goto LABEL_79;
  if ( v57 - (v71 + *(_DWORD *)(v5 + 908)) <= v57 >> 3 )
  {
    sub_1DE4DF0(v109, v57);
    v79 = *(_DWORD *)(v5 + 912);
    if ( v79 )
    {
      v80 = v79 - 1;
      v81 = *(_QWORD *)(v5 + 896);
      v75 = 0;
      v82 = 1;
      LODWORD(v83) = v80 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v71 = *(_DWORD *)(v5 + 904) + 1;
      v69 = (__int64 *)(v81 + 16LL * (unsigned int)v83);
      v84 = *v69;
      if ( v51 == *v69 )
        goto LABEL_73;
      while ( v84 != -8 )
      {
        if ( !v75 && v84 == -16 )
          v75 = v69;
        v83 = v80 & (unsigned int)(v83 + v82);
        v69 = (__int64 *)(v81 + 16 * v83);
        v84 = *v69;
        if ( *v69 == v51 )
          goto LABEL_73;
        ++v82;
      }
      goto LABEL_83;
    }
    goto LABEL_175;
  }
LABEL_73:
  *(_DWORD *)(v5 + 904) = v71;
  if ( *v69 != -8 )
    --*(_DWORD *)(v5 + 908);
  *v69 = v51;
  v69[1] = 0;
LABEL_76:
  v123 = (_QWORD *)v51;
  v124[0] = v51;
  result = *(_QWORD *)sub_1DE4FA0(v109, v124)[1];
  v19 = *(_QWORD *)result;
  if ( *(_QWORD *)result )
    goto LABEL_11;
  return result;
}
