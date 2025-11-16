// Function: sub_2E54D10
// Address: 0x2e54d10
//
__int64 __fastcall sub_2E54D10(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 v8; // rsi
  _QWORD *v9; // rdi
  unsigned int v10; // edx
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  __int64 v16; // rbx
  _QWORD *v17; // rax
  int v18; // eax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r13
  int *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  int *v26; // rdx
  int v27; // eax
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // rdi
  __int64 v30; // rsi
  unsigned int v31; // ecx
  __int64 v32; // rsi
  unsigned int v33; // edx
  int v34; // ebx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rax
  __int64 (__fastcall *v40)(__int64, __int64); // rax
  char v41; // al
  __int64 v42; // r8
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r13
  __int64 v46; // r12
  __int64 v47; // rcx
  __int64 v48; // r11
  unsigned int v49; // edi
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // r8
  unsigned int v53; // esi
  __int64 v54; // rdx
  __int64 v55; // rsi
  char v56; // al
  __int64 v57; // r11
  __int64 v58; // rcx
  __int64 v59; // rax
  bool v60; // cf
  __int64 v61; // r11
  __int64 v62; // rsi
  unsigned __int64 v63; // rax
  __int64 v64; // r9
  char v65; // al
  __int64 v66; // r11
  char v67; // al
  __int64 v68; // rdi
  unsigned __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // r11
  __int64 v72; // rcx
  __int64 v73; // rsi
  unsigned __int8 *v74; // rsi
  bool v75; // zf
  char v76; // al
  __int64 v77; // r11
  __int64 *v78; // rax
  unsigned int v79; // esi
  int v80; // eax
  __int64 *v81; // rdx
  int v82; // eax
  int v84; // ecx
  unsigned int v85; // esi
  __int64 *v86; // rax
  int v87; // edx
  __int64 v88; // rdx
  unsigned __int64 v89; // r13
  unsigned __int64 v90; // rdi
  char v91; // al
  unsigned __int64 v92; // rax
  char v93; // al
  unsigned int v94; // [rsp+0h] [rbp-260h]
  __int64 v95; // [rsp+0h] [rbp-260h]
  __int64 v96; // [rsp+0h] [rbp-260h]
  __int64 v97; // [rsp+10h] [rbp-250h]
  __int64 v98; // [rsp+10h] [rbp-250h]
  __int64 v99; // [rsp+10h] [rbp-250h]
  __int64 v100; // [rsp+10h] [rbp-250h]
  __int64 (__fastcall *v101)(__int64, __int64, unsigned __int64, __int64); // [rsp+10h] [rbp-250h]
  __int64 v102; // [rsp+10h] [rbp-250h]
  unsigned __int8 v103; // [rsp+1Fh] [rbp-241h]
  char v104; // [rsp+28h] [rbp-238h]
  __int64 v105; // [rsp+28h] [rbp-238h]
  __int64 v106; // [rsp+28h] [rbp-238h]
  __int64 v107; // [rsp+28h] [rbp-238h]
  __int64 v109; // [rsp+38h] [rbp-228h]
  __int64 v110; // [rsp+38h] [rbp-228h]
  __int64 v111; // [rsp+38h] [rbp-228h]
  unsigned int v112; // [rsp+38h] [rbp-228h]
  __int64 v113; // [rsp+38h] [rbp-228h]
  __int64 v114; // [rsp+38h] [rbp-228h]
  __int64 v115; // [rsp+40h] [rbp-220h]
  __int64 v116; // [rsp+40h] [rbp-220h]
  __int64 v117; // [rsp+40h] [rbp-220h]
  unsigned __int64 v118; // [rsp+40h] [rbp-220h]
  __int64 v119; // [rsp+40h] [rbp-220h]
  __int64 v120; // [rsp+40h] [rbp-220h]
  int *v121; // [rsp+48h] [rbp-218h]
  char v122; // [rsp+48h] [rbp-218h]
  __int64 v123; // [rsp+58h] [rbp-208h]
  __int64 v124; // [rsp+60h] [rbp-200h]
  __int64 v125; // [rsp+68h] [rbp-1F8h]
  char v126; // [rsp+77h] [rbp-1E9h] BYREF
  __int64 v127; // [rsp+78h] [rbp-1E8h] BYREF
  __int64 v128; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 *v129; // [rsp+88h] [rbp-1D8h] BYREF
  unsigned __int8 *v130; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 *v131; // [rsp+98h] [rbp-1C8h] BYREF
  __int64 *v132[2]; // [rsp+A0h] [rbp-1C0h] BYREF
  _BYTE v133[16]; // [rsp+B0h] [rbp-1B0h] BYREF
  _BYTE *v134; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 v135; // [rsp+C8h] [rbp-198h]
  _BYTE v136[40]; // [rsp+D0h] [rbp-190h] BYREF
  int v137; // [rsp+F8h] [rbp-168h] BYREF
  unsigned __int64 v138; // [rsp+100h] [rbp-160h]
  int *v139; // [rsp+108h] [rbp-158h]
  int *v140; // [rsp+110h] [rbp-150h]
  __int64 v141; // [rsp+118h] [rbp-148h]
  _BYTE *v142; // [rsp+120h] [rbp-140h] BYREF
  __int64 v143; // [rsp+128h] [rbp-138h]
  _BYTE v144[304]; // [rsp+130h] [rbp-130h] BYREF

  v7 = *((_DWORD *)a1 + 32);
  v142 = v144;
  v143 = 0x2000000000LL;
  ++a1[14];
  v123 = (__int64)(a1 + 14);
  if ( v7 )
  {
    v30 = *((unsigned int *)a1 + 34);
    a6 = 64;
    v9 = (_QWORD *)a1[15];
    v31 = 4 * v7;
    a5 = v30;
    v32 = 2 * v30;
    if ( (unsigned int)(4 * v7) < 0x40 )
      v31 = 64;
    v12 = &v9[v32];
    if ( v31 >= (unsigned int)a5 )
      goto LABEL_4;
    v33 = v7 - 1;
    if ( v33 )
    {
      _BitScanReverse(&v33, v33);
      v34 = 1 << (33 - (v33 ^ 0x1F));
      if ( v34 < 64 )
        v34 = 64;
      if ( v34 == (_DWORD)a5 )
      {
        a1[16] = 0;
        do
        {
          if ( v9 )
            *v9 = 0;
          v9 += 2;
        }
        while ( v9 != v12 );
LABEL_133:
        v13 = (unsigned int)v143;
        v14 = HIDWORD(v143);
        v15 = (unsigned int)v143 + 1LL;
LABEL_8:
        v16 = *(_QWORD *)(a2 + 96);
        if ( v15 > v14 )
        {
          sub_C8D5F0((__int64)&v142, v144, v15, 8u, a5, a6);
          v17 = &v142[8 * (unsigned int)v143];
        }
        else
        {
          v17 = &v142[8 * v13];
        }
        goto LABEL_10;
      }
    }
    else
    {
      v34 = 64;
    }
    sub_C7D6A0((__int64)v9, v32 * 8, 8);
    v35 = ((((((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
             | (4 * v34 / 3u + 1)
             | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
           | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 16;
    v36 = (v35
         | (((((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
             | (4 * v34 / 3u + 1)
             | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
           | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1))
        + 1;
    *((_DWORD *)a1 + 34) = v36;
    a1[15] = sub_C7D670(16 * v36, 8);
    sub_2E512C0(v123);
    v13 = (unsigned int)v143;
    v14 = HIDWORD(v143);
    v15 = (unsigned int)v143 + 1LL;
    goto LABEL_8;
  }
  if ( *((_DWORD *)a1 + 33) )
  {
    v8 = *((unsigned int *)a1 + 34);
    v9 = (_QWORD *)a1[15];
    v10 = v8;
    v11 = 2 * v8;
    v12 = &v9[v11];
    if ( v10 <= 0x40 )
    {
LABEL_4:
      if ( v12 == v9 )
      {
        v14 = 32;
        v15 = 1;
        v13 = 0;
      }
      else
      {
        do
        {
          *v9 = 0;
          v9 += 2;
        }
        while ( v9 != v12 );
        v13 = (unsigned int)v143;
        v14 = HIDWORD(v143);
        v15 = (unsigned int)v143 + 1LL;
      }
      a1[16] = 0;
      goto LABEL_8;
    }
    sub_C7D6A0((__int64)v9, v11 * 8, 8);
    a1[15] = 0;
    a1[16] = 0;
    *((_DWORD *)a1 + 34) = 0;
    goto LABEL_133;
  }
  v16 = *(_QWORD *)(a2 + 96);
  v17 = v144;
LABEL_10:
  *v17 = v16;
  v103 = 0;
  v18 = v143 + 1;
  LODWORD(v143) = v143 + 1;
  do
  {
    v19 = *(_QWORD *)&v142[8 * v18 - 8];
    LODWORD(v143) = v18 - 1;
    sub_2E4F180(
      (__int64)&v142,
      &v142[8 * (v18 - 1)],
      *(char **)(v19 + 24),
      (char *)(*(_QWORD *)(v19 + 24) + 8LL * *(unsigned int *)(v19 + 32)));
    v20 = *(_QWORD *)v19;
    a1[8] = 0;
    v104 = 0;
    v124 = v20;
    v125 = v20 + 48;
    if ( v20 + 48 == *(_QWORD *)(v20 + 56) )
      goto LABEL_101;
    v21 = *(_QWORD *)(v20 + 56);
    v22 = &v137;
    while ( 1 )
    {
      if ( !v21 )
        BUG();
      v23 = v21;
      if ( (*(_BYTE *)v21 & 4) == 0 && (*(_BYTE *)(v21 + 44) & 8) != 0 )
      {
        do
          v23 = *(_QWORD *)(v23 + 8);
        while ( (*(_BYTE *)(v23 + 44) & 8) != 0 );
      }
      v24 = *(_QWORD *)(v23 + 8);
      v134 = v136;
      v135 = 0x800000000LL;
      v137 = 0;
      v138 = 0;
      v139 = v22;
      v140 = v22;
      v141 = 0;
      if ( !sub_2E501D0(v21) )
        goto LABEL_20;
      v25 = *(_QWORD *)(v21 + 48);
      v26 = (int *)(v25 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v25 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v27 = v25 & 7;
        switch ( v27 )
        {
          case 1:
            goto LABEL_20;
          case 3:
            v37 = *((unsigned __int8 *)v26 + 4);
            if ( (_BYTE)v37 && *(_QWORD *)&v26[2 * *v26 + 4]
              || *((_BYTE *)v26 + 5) && *(_QWORD *)&v26[2 * *v26 + 4 + 2 * v37] )
            {
              goto LABEL_20;
            }
            break;
          case 2:
            goto LABEL_20;
        }
      }
      if ( (unsigned __int8)sub_2E50190(v21, 23, 1)
        || (unsigned int)*(unsigned __int16 *)(v21 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(v21 + 32) + 64LL) & 8) != 0 )
      {
        goto LABEL_20;
      }
      v38 = *(_DWORD *)(v21 + 44);
      if ( (v38 & 4) != 0 || (v38 & 8) == 0 )
        v39 = (*(_QWORD *)(*(_QWORD *)(v21 + 16) + 24LL) >> 19) & 1LL;
      else
        LOBYTE(v39) = sub_2E88A90(v21, 0x80000, 1);
      if ( (_BYTE)v39 )
        goto LABEL_20;
      v40 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 176LL);
      v41 = v40 == sub_2E4F5F0 ? sub_2E50190(v21, 30, 2) : v40(*a1, v21);
      if ( v41
        || (unsigned int)sub_2E88FE0(v21) + *(unsigned __int8 *)(*(_QWORD *)(v21 + 16) + 9LL) != 1
        || (unsigned int)sub_2E88FE0(v21) != 1 )
      {
        goto LABEL_20;
      }
      v43 = *(_QWORD *)(v21 + 32);
      v44 = v43 + 40LL * (*(_DWORD *)(v21 + 40) & 0xFFFFFF);
      if ( v43 != v44 )
      {
        v115 = v21;
        v45 = v43 + 40LL * (*(_DWORD *)(v21 + 40) & 0xFFFFFF);
        v121 = v22;
        v46 = v43;
        while ( 1 )
        {
          if ( !*(_BYTE *)v46 && *(int *)(v46 + 8) >= 0 )
          {
            if ( (*(_BYTE *)(v46 + 3) & 0x10) != 0 )
            {
              v22 = v121;
              goto LABEL_20;
            }
            LODWORD(v131) = *(_DWORD *)(v46 + 8);
            sub_2E52950((__int64)v132, (__int64)&v134, (unsigned int *)&v131, v44, v42);
          }
          v46 += 40;
          if ( v45 == v46 )
          {
            v21 = v115;
            v22 = v121;
            break;
          }
        }
      }
      v130 = (unsigned __int8 *)v21;
      if ( !(unsigned __int8)sub_2E51140(v123, (__int64 *)&v130, &v131) )
      {
        v79 = *((_DWORD *)a1 + 34);
        v80 = *((_DWORD *)a1 + 32);
        v81 = v131;
        ++a1[14];
        v82 = v80 + 1;
        v132[0] = v81;
        if ( 4 * v82 >= 3 * v79 )
        {
          v79 *= 2;
        }
        else if ( v79 - *((_DWORD *)a1 + 33) - v82 > v79 >> 3 )
        {
          goto LABEL_96;
        }
        sub_2E52190(v123, v79);
        sub_2E51140(v123, (__int64 *)&v130, v132);
        v81 = v132[0];
        v82 = *((_DWORD *)a1 + 32) + 1;
LABEL_96:
        *((_DWORD *)a1 + 32) = v82;
        if ( *v81 )
          --*((_DWORD *)a1 + 33);
        *v81 = (__int64)v130;
        v81[1] = v124;
LABEL_20:
        v28 = v138;
        while ( v28 )
        {
          sub_2E4F620(*(_QWORD *)(v28 + 24));
          v29 = v28;
          v28 = *(_QWORD *)(v28 + 16);
          j_j___libc_free_0(v29);
        }
        goto LABEL_22;
      }
      v47 = v131[1];
      v48 = *(_QWORD *)(*(_QWORD *)(v124 + 32) + 328LL);
      if ( v124 != v48 && v47 != v48 )
      {
        v49 = *(_DWORD *)(a2 + 32);
        v50 = (unsigned int)(*(_DWORD *)(v124 + 24) + 1);
        v51 = 0;
        if ( (unsigned int)v50 < v49 )
          v51 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v50);
        if ( v47 )
        {
          v52 = (unsigned int)(*(_DWORD *)(v47 + 24) + 1);
          v53 = *(_DWORD *)(v47 + 24) + 1;
        }
        else
        {
          v52 = 0;
          v53 = 0;
        }
        v54 = 0;
        if ( v49 > v53 )
          v54 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v52);
        for ( ; v51 != v54; v51 = *(_QWORD *)(v51 + 8) )
        {
          if ( *(_DWORD *)(v51 + 16) < *(_DWORD *)(v54 + 16) )
          {
            v55 = v51;
            v51 = v54;
            v54 = v55;
          }
        }
        v48 = *(_QWORD *)v54;
      }
      v109 = v131[1];
      v116 = v48;
      if ( !(unsigned __int8)sub_2E31B00(v48) )
        goto LABEL_20;
      v56 = sub_B2D610(**(_QWORD **)(v116 + 32), 18);
      v57 = v116;
      v58 = v109;
      if ( !v56 )
      {
        v97 = v116;
        v117 = sub_2E39EA0((__int64 *)a1[4], v109);
        v59 = sub_2E39EA0((__int64 *)a1[4], v124);
        v60 = __CFADD__(v117, v59);
        v118 = v117 + v59;
        v61 = v97;
        if ( v60 )
        {
          sub_2E39EA0((__int64 *)a1[4], v97);
          v58 = v109;
          v57 = v97;
        }
        else
        {
          v62 = v97;
          v98 = v109;
          v110 = v61;
          v63 = sub_2E39EA0((__int64 *)a1[4], v62);
          v57 = v110;
          v58 = v98;
          if ( v118 < v63 )
            goto LABEL_20;
        }
      }
      v119 = v57;
      if ( v58 != v57 )
      {
        v64 = *(_QWORD *)(v58 + 16);
        if ( *(_QWORD *)(v124 + 16) )
        {
          if ( v64 )
          {
            v99 = *(_QWORD *)(v124 + 16);
            v111 = *(_QWORD *)(v58 + 16);
            v65 = sub_D0E9D0(v64, v99, 0, 0, 0);
            v66 = v119;
            if ( v65 || (v67 = sub_D0E9D0(v99, v111, 0, 0, 0), v66 = v119, v67) )
            {
              if ( (unsigned int)*(unsigned __int16 *)(v21 + 68) - 1 <= 1
                && (*(_BYTE *)(*(_QWORD *)(v21 + 32) + 64LL) & 0x20) != 0
                || (*(_BYTE *)(v21 + 46) & 2) == 0 && (v120 = v66, v91 = sub_2E50190(v21, 36, 1), v66 = v120, v91) )
              {
                if ( v124 != v66 )
                  goto LABEL_20;
              }
              v132[0] = (__int64 *)v133;
              v132[1] = (__int64 *)0x200000000LL;
              if ( (_DWORD)v135 || v141 )
              {
                v113 = v66;
                v92 = sub_2E313E0(v66);
                v93 = sub_2E4F9C0((__int64)a1, v92, v21, (__int64)&v134, (__int64 *)v132, &v126);
                v66 = v113;
                if ( !v93 )
                  goto LABEL_122;
              }
              v100 = v66;
              v94 = *(_DWORD *)(*(_QWORD *)(v21 + 32) + 8LL);
              v112 = sub_2EC0780(a1[3], v94, byte_3F871B3, 0);
              v131 = 0;
              v122 = sub_2E52EE0(a1, v112, v94, v100, v21, &v131);
              if ( !v122 )
              {
LABEL_122:
                if ( (_BYTE *)v132[0] != v133 )
                  _libc_free((unsigned __int64)v132[0]);
                goto LABEL_20;
              }
              v68 = v100;
              v105 = v100;
              v95 = *a1;
              v101 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, __int64))(*(_QWORD *)*a1 + 248LL);
              v69 = sub_2E313E0(v68);
              v70 = v101(v95, v105, v69, v21);
              v71 = v105;
              v127 = 0;
              v72 = v70;
              v130 = 0;
              if ( (unsigned __int8 **)(v70 + 56) != &v130 )
              {
                v73 = *(_QWORD *)(v70 + 56);
                if ( v73 )
                {
                  v102 = v105;
                  v106 = v70;
                  v96 = v70 + 56;
                  sub_B91220(v70 + 56, v73);
                  v74 = v130;
                  v72 = v106;
                  v71 = v102;
                  v75 = v130 == 0;
                  *(_QWORD *)(v106 + 56) = v130;
                  if ( !v75 )
                  {
                    sub_B976B0((__int64)&v130, v74, v96);
                    v72 = v106;
                    v71 = v102;
                  }
                }
              }
              v107 = v71;
              sub_2EAB0C0(*(_QWORD *)(v72 + 32), v112);
              v128 = v21;
              v76 = sub_2E51140(v123, &v128, &v129);
              v77 = v107;
              if ( v76 )
              {
                v78 = v129 + 1;
LABEL_110:
                *v78 = v77;
                if ( v127 )
                  sub_B91220((__int64)&v127, v127);
                if ( (_BYTE *)v132[0] != v133 )
                  _libc_free((unsigned __int64)v132[0]);
                v104 = v122;
                goto LABEL_115;
              }
              v84 = *((_DWORD *)a1 + 32);
              v85 = *((_DWORD *)a1 + 34);
              v86 = v129;
              ++a1[14];
              v87 = v84 + 1;
              v130 = (unsigned __int8 *)v86;
              if ( 4 * (v84 + 1) >= 3 * v85 )
              {
                v114 = v107;
                v85 *= 2;
              }
              else
              {
                if ( v85 - *((_DWORD *)a1 + 33) - v87 > v85 >> 3 )
                {
LABEL_107:
                  *((_DWORD *)a1 + 32) = v87;
                  if ( *v86 )
                    --*((_DWORD *)a1 + 33);
                  v88 = v128;
                  v86[1] = 0;
                  v78 = v86 + 1;
                  *(v78 - 1) = v88;
                  goto LABEL_110;
                }
                v114 = v107;
              }
              sub_2E52190(v123, v85);
              sub_2E51140(v123, &v128, (__int64 **)&v130);
              v77 = v114;
              v87 = *((_DWORD *)a1 + 32) + 1;
              v86 = (__int64 *)v130;
              goto LABEL_107;
            }
          }
        }
      }
LABEL_115:
      v89 = v138;
      while ( v89 )
      {
        sub_2E4F620(*(_QWORD *)(v89 + 24));
        v90 = v89;
        v89 = *(_QWORD *)(v89 + 16);
        j_j___libc_free_0(v90);
      }
LABEL_22:
      if ( v134 != v136 )
        _libc_free((unsigned __int64)v134);
      if ( v24 == v125 )
        break;
      v21 = v24;
    }
    v103 |= v104;
LABEL_101:
    v18 = v143;
  }
  while ( (_DWORD)v143 );
  if ( v142 != v144 )
    _libc_free((unsigned __int64)v142);
  return v103;
}
