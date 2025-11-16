// Function: sub_D3D0C0
// Address: 0xd3d0c0
//
__int64 __fastcall sub_D3D0C0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        unsigned __int64 a6)
{
  __int64 v6; // r12
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 *v11; // rax
  __int64 *v12; // rdx
  _QWORD *v13; // rax
  unsigned __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rdi
  _QWORD *v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // rdx
  __int64 v22; // rsi
  __int64 *v23; // rax
  int v24; // r10d
  unsigned __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 **v27; // rax
  __int64 **v28; // r11
  unsigned int *v29; // rbx
  __int64 **v30; // rcx
  int v31; // r11d
  __int64 v32; // r10
  _QWORD *v33; // rdi
  unsigned __int64 v34; // rdx
  unsigned int v35; // ecx
  _QWORD *v36; // rax
  __int64 v37; // r8
  unsigned int *v38; // rax
  unsigned int v39; // edi
  __int64 *v40; // rcx
  unsigned int *v41; // r13
  __int64 *v42; // rcx
  __int64 *v43; // rsi
  unsigned int v44; // ebx
  int v45; // r14d
  unsigned int v46; // eax
  unsigned int v47; // eax
  _QWORD *v48; // r14
  __int64 v49; // rcx
  unsigned int *v50; // rcx
  unsigned int v51; // eax
  __int64 v53; // rax
  int v54; // r8d
  int v55; // r8d
  __int64 v56; // r10
  unsigned __int64 v57; // r9
  int v58; // edx
  unsigned int v59; // eax
  _QWORD *v60; // rcx
  __int64 v61; // rbx
  int v62; // edi
  _QWORD *v63; // rsi
  int v64; // eax
  int v65; // eax
  __int64 v66; // rax
  int v67; // edx
  __int64 v68; // r13
  unsigned __int64 v69; // r11
  int v70; // edx
  unsigned int v71; // ecx
  __int64 v72; // r10
  __int64 v73; // rdx
  int v74; // edi
  int v75; // edx
  __int64 v76; // r13
  int v77; // edi
  unsigned __int64 v78; // r11
  unsigned int v79; // ecx
  __int64 v80; // r10
  int v81; // r13d
  int v82; // r13d
  unsigned __int64 v83; // r10
  unsigned int v84; // edx
  __int64 v85; // r8
  int v86; // esi
  _QWORD *v87; // rcx
  int v88; // r13d
  int v89; // r13d
  int v90; // esi
  unsigned __int64 v91; // r10
  unsigned int v92; // edx
  __int64 v93; // r8
  int v94; // eax
  int v95; // eax
  __int64 *v96; // rax
  __int64 *v97; // rbx
  _QWORD *v98; // r12
  __int64 v99; // r14
  int v100; // r13d
  int v101; // eax
  __int64 v102; // rax
  int v103; // r12d
  __int64 v104; // r10
  unsigned int v105; // edx
  int v106; // edi
  _QWORD *v107; // rdx
  _QWORD *v108; // rcx
  _BYTE *v109; // rdi
  _BYTE *v110; // rax
  int v111; // edi
  int v112; // r8d
  int v113; // r8d
  __int64 v114; // r10
  int v115; // edi
  unsigned __int64 v116; // r9
  unsigned int v117; // eax
  __int64 v118; // rbx
  int v119; // r12d
  __int64 v120; // r10
  int v121; // edi
  unsigned int v122; // edx
  int v123; // ecx
  __int64 **v124; // r11
  __int64 v125; // [rsp+8h] [rbp-108h]
  unsigned __int64 v126; // [rsp+18h] [rbp-F8h]
  __int64 *v127; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v128; // [rsp+28h] [rbp-E8h]
  unsigned int *v129; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v130; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v131; // [rsp+40h] [rbp-D0h]
  __int64 *v132; // [rsp+48h] [rbp-C8h]
  unsigned int *v133; // [rsp+60h] [rbp-B0h]
  unsigned int *v134; // [rsp+60h] [rbp-B0h]
  unsigned int *v135; // [rsp+60h] [rbp-B0h]
  _QWORD *v136; // [rsp+60h] [rbp-B0h]
  unsigned int *v137; // [rsp+68h] [rbp-A8h]
  _QWORD *v138; // [rsp+68h] [rbp-A8h]
  __int64 v139; // [rsp+80h] [rbp-90h] BYREF
  __int64 *v140; // [rsp+88h] [rbp-88h]
  __int64 v141; // [rsp+90h] [rbp-80h]
  int v142; // [rsp+98h] [rbp-78h]
  unsigned __int8 v143; // [rsp+9Ch] [rbp-74h]
  char v144; // [rsp+A0h] [rbp-70h] BYREF

  v140 = (__int64 *)&v144;
  *(_QWORD *)(a1 + 208) = -1;
  v8 = *(__int64 **)a3;
  v9 = *(unsigned int *)(a3 + 8);
  v126 = a2;
  v139 = 0;
  v141 = 8;
  v142 = 0;
  v143 = 1;
  v127 = &v8[v9];
  if ( v8 == v127 )
  {
    LOBYTE(v6) = *(_DWORD *)(a1 + 228) == 0;
    return (unsigned int)v6;
  }
  v132 = v8;
  LOBYTE(v10) = 1;
  v128 = a2 + 8;
  do
  {
    v6 = *v132;
    if ( (_BYTE)v10 )
    {
      v11 = v140;
      v12 = &v140[HIDWORD(v141)];
      if ( v140 != v12 )
      {
        do
        {
          if ( v6 == *v11 )
            goto LABEL_8;
          ++v11;
        }
        while ( v12 != v11 );
      }
    }
    else
    {
      a2 = *v132;
      if ( sub_C8CA60((__int64)&v139, *v132) )
      {
        LOBYTE(v10) = v143;
        goto LABEL_8;
      }
    }
    v13 = *(_QWORD **)(v126 + 16);
    if ( !v13 )
      goto LABEL_63;
    v14 = v128;
    v15 = *(_QWORD *)(v126 + 16);
    do
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v15 + 16);
        v17 = *(_QWORD *)(v15 + 24);
        if ( *(_QWORD *)(v15 + 48) >= v6 )
          break;
        v15 = *(_QWORD *)(v15 + 24);
        if ( !v17 )
          goto LABEL_15;
      }
      v14 = v15;
      v15 = *(_QWORD *)(v15 + 16);
    }
    while ( v16 );
LABEL_15:
    if ( v128 == v14 )
      goto LABEL_17;
    if ( *(_QWORD *)(v14 + 48) > v6 )
      goto LABEL_17;
    v15 = v14 + 32;
    if ( (*(_BYTE *)(v14 + 40) & 1) != 0 )
      goto LABEL_17;
    v15 = *(_QWORD *)(v14 + 32);
    if ( (*(_BYTE *)(v15 + 8) & 1) != 0 )
      goto LABEL_17;
    v6 = *(_QWORD *)v15;
    if ( (*(_BYTE *)(*(_QWORD *)v15 + 8LL) & 1) != 0 )
    {
      v15 = *(_QWORD *)v15;
    }
    else
    {
      v48 = *(_QWORD **)v6;
      if ( (*(_BYTE *)(*(_QWORD *)v6 + 8LL) & 1) == 0 )
      {
        v107 = (_QWORD *)*v48;
        if ( (*(_BYTE *)(*v48 + 8LL) & 1) != 0 )
        {
          v48 = (_QWORD *)*v48;
        }
        else
        {
          v108 = (_QWORD *)*v107;
          if ( (*(_BYTE *)(*v107 + 8LL) & 1) == 0 )
          {
            v109 = (_BYTE *)*v108;
            v136 = (_QWORD *)*v107;
            if ( (*(_BYTE *)(*v108 + 8LL) & 1) == 0 )
            {
              v138 = (_QWORD *)*v48;
              v110 = sub_D38E40(v109);
              v107 = v138;
              v109 = v110;
              *v136 = v110;
            }
            *v107 = v109;
            v108 = v109;
          }
          *v48 = v108;
          v48 = v108;
        }
        *(_QWORD *)v6 = v48;
      }
      *(_QWORD *)v15 = v48;
      v15 = (__int64)v48;
    }
    *(_QWORD *)(v14 + 32) = v15;
    v13 = *(_QWORD **)(v126 + 16);
    if ( v13 )
    {
LABEL_17:
      v18 = *(_QWORD *)(v15 + 16);
      v19 = (_QWORD *)v128;
      do
      {
        while ( 1 )
        {
          a2 = v13[2];
          v20 = v13[3];
          if ( v13[6] >= v18 )
            break;
          v13 = (_QWORD *)v13[3];
          if ( !v20 )
            goto LABEL_21;
        }
        v19 = v13;
        v13 = (_QWORD *)v13[2];
      }
      while ( a2 );
LABEL_21:
      if ( (_QWORD *)v128 != v19 && v18 < v19[6] )
        v19 = (_QWORD *)v128;
    }
    else
    {
LABEL_63:
      v19 = (_QWORD *)v128;
    }
    v10 = v143;
    if ( (v19[5] & 1) == 0 )
      goto LABEL_8;
    v130 = (unsigned __int64)(v19 + 4);
    v125 = a1 + 24;
    do
    {
      v21 = (__int64 *)(v130 + 16);
      v22 = *(_QWORD *)(v130 + 16);
      if ( !(_BYTE)v10 )
      {
LABEL_84:
        sub_C8CC70((__int64)&v139, v22, (__int64)v21, v10, (__int64)a5, a6);
        LOBYTE(v10) = v143;
        v22 = *(_QWORD *)(v130 + 16);
        goto LABEL_31;
      }
      v23 = v140;
      v21 = &v140[HIDWORD(v141)];
      if ( v140 == v21 )
      {
LABEL_83:
        if ( HIDWORD(v141) >= (unsigned int)v141 )
          goto LABEL_84;
        ++HIDWORD(v141);
        *v21 = v22;
        v22 = *(_QWORD *)(v130 + 16);
        ++v139;
        LOBYTE(v10) = v143;
      }
      else
      {
        while ( v22 != *v23 )
        {
          if ( v21 == ++v23 )
            goto LABEL_83;
        }
      }
LABEL_31:
      a2 = v22 & 4;
      v131 = v130;
      if ( !(_DWORD)a2 )
      {
        v131 = *(_QWORD *)(v130 + 8) & 0xFFFFFFFFFFFFFFFELL;
        if ( !v131 )
          break;
        a2 = *(unsigned int *)(a1 + 48);
        if ( !(_DWORD)a2 )
          goto LABEL_87;
        goto LABEL_33;
      }
      do
      {
        a2 = *(unsigned int *)(a1 + 48);
        if ( !(_DWORD)a2 )
        {
LABEL_87:
          ++*(_QWORD *)(a1 + 24);
          goto LABEL_88;
        }
LABEL_33:
        v24 = a2 - 1;
        a6 = *(_QWORD *)(a1 + 32);
        v25 = *(_QWORD *)(v130 + 16);
        v26 = (a2 - 1) & (v25 ^ (v25 >> 9));
        v27 = (__int64 **)(a6 + 32LL * v26);
        a5 = *v27;
        v28 = v27;
        if ( (__int64 *)v25 == *v27 )
        {
LABEL_34:
          v29 = (unsigned int *)v28[1];
          goto LABEL_35;
        }
        v97 = *v27;
        v98 = (_QWORD *)(a6 + 32LL * (v24 & ((unsigned int)v25 ^ (unsigned int)(*(_QWORD *)(v130 + 16) >> 9))));
        LODWORD(v99) = (a2 - 1) & (v25 ^ (*(_QWORD *)(v130 + 16) >> 9));
        v100 = 1;
        v60 = 0;
        while ( v97 != (__int64 *)-4LL )
        {
          if ( v60 || v97 != (__int64 *)-16LL )
            v98 = v60;
          v99 = v24 & (unsigned int)(v99 + v100);
          v28 = (__int64 **)(a6 + 32 * v99);
          v97 = *v28;
          if ( (__int64 *)v25 == *v28 )
            goto LABEL_34;
          ++v100;
          v60 = v98;
          v98 = (_QWORD *)(a6 + 32 * v99);
        }
        v101 = *(_DWORD *)(a1 + 40);
        if ( !v60 )
          v60 = v98;
        ++*(_QWORD *)(a1 + 24);
        v58 = v101 + 1;
        if ( 4 * (v101 + 1) < (unsigned int)(3 * a2) )
        {
          if ( (int)a2 - *(_DWORD *)(a1 + 44) - v58 > (unsigned int)a2 >> 3 )
            goto LABEL_164;
          sub_D3A5D0(v125, a2);
          v112 = *(_DWORD *)(a1 + 48);
          if ( !v112 )
          {
LABEL_239:
            ++*(_DWORD *)(a1 + 40);
            BUG();
          }
          v113 = v112 - 1;
          v114 = *(_QWORD *)(a1 + 32);
          v63 = 0;
          v58 = *(_DWORD *)(a1 + 40) + 1;
          v115 = 1;
          v116 = *(_QWORD *)(v130 + 16);
          v117 = v113 & (v116 ^ (v116 >> 9));
          v60 = (_QWORD *)(v114 + 32LL * v117);
          v118 = *v60;
          if ( v116 == *v60 )
            goto LABEL_164;
          while ( v118 != -4 )
          {
            if ( v118 == -16 && !v63 )
              v63 = v60;
            v117 = v113 & (v115 + v117);
            v60 = (_QWORD *)(v114 + 32LL * v117);
            v118 = *v60;
            if ( v116 == *v60 )
              goto LABEL_164;
            ++v115;
          }
          goto LABEL_92;
        }
LABEL_88:
        sub_D3A5D0(v125, 2 * a2);
        v54 = *(_DWORD *)(a1 + 48);
        if ( !v54 )
          goto LABEL_239;
        v55 = v54 - 1;
        v56 = *(_QWORD *)(a1 + 32);
        v57 = *(_QWORD *)(v130 + 16);
        v58 = *(_DWORD *)(a1 + 40) + 1;
        v59 = v55 & (v57 ^ (v57 >> 9));
        v60 = (_QWORD *)(v56 + 32LL * v59);
        v61 = *v60;
        if ( v57 == *v60 )
          goto LABEL_164;
        v62 = 1;
        v63 = 0;
        while ( v61 != -4 )
        {
          if ( v61 == -16 && !v63 )
            v63 = v60;
          v59 = v55 & (v62 + v59);
          v60 = (_QWORD *)(v56 + 32LL * v59);
          v61 = *v60;
          if ( v57 == *v60 )
            goto LABEL_164;
          ++v62;
        }
LABEL_92:
        if ( v63 )
          v60 = v63;
LABEL_164:
        *(_DWORD *)(a1 + 40) = v58;
        if ( *v60 != -4 )
          --*(_DWORD *)(a1 + 44);
        v102 = *(_QWORD *)(v130 + 16);
        v60[1] = 0;
        v60[2] = 0;
        *v60 = v102;
        v60[3] = 0;
        a2 = *(unsigned int *)(a1 + 48);
        if ( !(_DWORD)a2 )
        {
          ++*(_QWORD *)(a1 + 24);
          v29 = 0;
          goto LABEL_168;
        }
        v24 = a2 - 1;
        a6 = *(_QWORD *)(a1 + 32);
        v29 = 0;
        v25 = *(_QWORD *)(v130 + 16);
        v26 = (a2 - 1) & (v25 ^ (v25 >> 9));
        v27 = (__int64 **)(a6 + 32LL * v26);
        a5 = *v27;
LABEL_35:
        LODWORD(v6) = 1;
        v30 = 0;
        if ( (__int64 *)v25 == a5 )
          goto LABEL_36;
        while ( 1 )
        {
          if ( a5 == (__int64 *)-4LL )
          {
            if ( !v30 )
              v30 = v27;
            v94 = *(_DWORD *)(a1 + 40);
            ++*(_QWORD *)(a1 + 24);
            v95 = v94 + 1;
            if ( 4 * v95 < (unsigned int)(3 * a2) )
            {
              if ( (int)a2 - (v95 + *(_DWORD *)(a1 + 44)) > (unsigned int)a2 >> 3 )
              {
LABEL_155:
                *(_DWORD *)(a1 + 40) = v95;
                if ( *v30 != (__int64 *)-4LL )
                  --*(_DWORD *)(a1 + 44);
                v129 = 0;
                v96 = *(__int64 **)(v130 + 16);
                v30[1] = 0;
                v30[2] = 0;
                *v30 = v96;
                v30[3] = 0;
                goto LABEL_37;
              }
              sub_D3A5D0(v125, a2);
              v119 = *(_DWORD *)(a1 + 48);
              if ( v119 )
              {
                LODWORD(v6) = v119 - 1;
                v120 = *(_QWORD *)(a1 + 32);
                v121 = 1;
                a2 = 0;
                a6 = *(_QWORD *)(v130 + 16);
                v95 = *(_DWORD *)(a1 + 40) + 1;
                v122 = v6 & (a6 ^ (a6 >> 9));
                v30 = (__int64 **)(v120 + 32LL * v122);
                a5 = *v30;
                if ( *v30 == (__int64 *)a6 )
                  goto LABEL_155;
                while ( a5 != (__int64 *)-4LL )
                {
                  if ( !a2 && a5 == (__int64 *)-16LL )
                    a2 = (unsigned __int64)v30;
                  v122 = v6 & (v121 + v122);
                  v30 = (__int64 **)(v120 + 32LL * v122);
                  a5 = *v30;
                  if ( (__int64 *)a6 == *v30 )
                    goto LABEL_155;
                  ++v121;
                }
                goto LABEL_172;
              }
              goto LABEL_240;
            }
LABEL_168:
            a2 = (unsigned int)(2 * a2);
            sub_D3A5D0(v125, a2);
            v103 = *(_DWORD *)(a1 + 48);
            if ( v103 )
            {
              LODWORD(v6) = v103 - 1;
              v104 = *(_QWORD *)(a1 + 32);
              a6 = *(_QWORD *)(v130 + 16);
              v95 = *(_DWORD *)(a1 + 40) + 1;
              v105 = v6 & (a6 ^ (a6 >> 9));
              v30 = (__int64 **)(v104 + 32LL * v105);
              a5 = *v30;
              if ( (__int64 *)a6 == *v30 )
                goto LABEL_155;
              v106 = 1;
              a2 = 0;
              while ( a5 != (__int64 *)-4LL )
              {
                if ( a5 == (__int64 *)-16LL && !a2 )
                  a2 = (unsigned __int64)v30;
                v105 = v6 & (v106 + v105);
                v30 = (__int64 **)(v104 + 32LL * v105);
                a5 = *v30;
                if ( (__int64 *)a6 == *v30 )
                  goto LABEL_155;
                ++v106;
              }
LABEL_172:
              if ( a2 )
                v30 = (__int64 **)a2;
              goto LABEL_155;
            }
LABEL_240:
            ++*(_DWORD *)(a1 + 40);
            BUG();
          }
          if ( v30 || a5 != (__int64 *)-16LL )
            v27 = v30;
          v123 = v6 + 1;
          LODWORD(v6) = v26 + v6;
          v26 = v24 & v6;
          v124 = (__int64 **)(a6 + 32LL * (v24 & (unsigned int)v6));
          a5 = *v124;
          if ( (__int64 *)v25 == *v124 )
            break;
          LODWORD(v6) = v123;
          v30 = v27;
          v27 = v124;
        }
        v27 = (__int64 **)(a6 + 32LL * (v24 & (unsigned int)v6));
LABEL_36:
        v129 = (unsigned int *)v27[2];
LABEL_37:
        v137 = v29;
        if ( v29 == v129 )
          goto LABEL_72;
        while ( 2 )
        {
          if ( v131 == v130 )
          {
            v38 = v137 + 1;
            v133 = v129;
          }
          else
          {
            a2 = *(unsigned int *)(a1 + 48);
            if ( (_DWORD)a2 )
            {
              v31 = a2 - 1;
              v32 = *(_QWORD *)(a1 + 32);
              LODWORD(v6) = 1;
              v33 = 0;
              v34 = *(_QWORD *)(v131 + 16);
              v35 = (a2 - 1) & (v34 ^ (v34 >> 9));
              v36 = (_QWORD *)(v32 + 32LL * v35);
              v37 = *v36;
              if ( *v36 == v34 )
              {
LABEL_41:
                v38 = (unsigned int *)v36[1];
                goto LABEL_42;
              }
              while ( v37 != -4 )
              {
                if ( !v33 && v37 == -16 )
                  v33 = v36;
                v35 = v31 & (v6 + v35);
                v36 = (_QWORD *)(v32 + 32LL * v35);
                v37 = *v36;
                if ( v34 == *v36 )
                  goto LABEL_41;
                LODWORD(v6) = v6 + 1;
              }
              if ( !v33 )
                v33 = v36;
              v64 = *(_DWORD *)(a1 + 40);
              ++*(_QWORD *)(a1 + 24);
              v65 = v64 + 1;
              if ( 4 * v65 < (unsigned int)(3 * a2) )
              {
                if ( (int)a2 - *(_DWORD *)(a1 + 44) - v65 > (unsigned int)a2 >> 3 )
                  goto LABEL_106;
                sub_D3A5D0(v125, a2);
                v88 = *(_DWORD *)(a1 + 48);
                if ( v88 )
                {
                  v89 = v88 - 1;
                  v6 = *(_QWORD *)(a1 + 32);
                  v90 = 1;
                  v87 = 0;
                  v91 = *(_QWORD *)(v131 + 16);
                  v65 = *(_DWORD *)(a1 + 40) + 1;
                  v92 = v89 & (v91 ^ (v91 >> 9));
                  v33 = (_QWORD *)(v6 + 32LL * v92);
                  v93 = *v33;
                  if ( *v33 != v91 )
                  {
                    while ( v93 != -4 )
                    {
                      if ( !v87 && v93 == -16 )
                        v87 = v33;
                      v92 = v89 & (v90 + v92);
                      v33 = (_QWORD *)(v6 + 32LL * v92);
                      v93 = *v33;
                      if ( v91 == *v33 )
                        goto LABEL_106;
                      ++v90;
                    }
                    goto LABEL_145;
                  }
                  goto LABEL_106;
                }
LABEL_241:
                ++*(_DWORD *)(a1 + 40);
                BUG();
              }
            }
            else
            {
              ++*(_QWORD *)(a1 + 24);
            }
            sub_D3A5D0(v125, 2 * a2);
            v81 = *(_DWORD *)(a1 + 48);
            if ( !v81 )
              goto LABEL_241;
            v82 = v81 - 1;
            v6 = *(_QWORD *)(a1 + 32);
            v83 = *(_QWORD *)(v131 + 16);
            v65 = *(_DWORD *)(a1 + 40) + 1;
            v84 = v82 & (v83 ^ (v83 >> 9));
            v33 = (_QWORD *)(v6 + 32LL * v84);
            v85 = *v33;
            if ( v83 != *v33 )
            {
              v86 = 1;
              v87 = 0;
              while ( v85 != -4 )
              {
                if ( v85 == -16 && !v87 )
                  v87 = v33;
                v84 = v82 & (v86 + v84);
                v33 = (_QWORD *)(v6 + 32LL * v84);
                v85 = *v33;
                if ( v83 == *v33 )
                  goto LABEL_106;
                ++v86;
              }
LABEL_145:
              if ( v87 )
                v33 = v87;
            }
LABEL_106:
            *(_DWORD *)(a1 + 40) = v65;
            if ( *v33 != -4 )
              --*(_DWORD *)(a1 + 44);
            v66 = *(_QWORD *)(v131 + 16);
            v33[1] = 0;
            v33[2] = 0;
            *v33 = v66;
            v33[3] = 0;
            a2 = *(unsigned int *)(a1 + 48);
            v32 = *(_QWORD *)(a1 + 32);
            if ( !(_DWORD)a2 )
            {
              ++*(_QWORD *)(a1 + 24);
              v38 = 0;
              goto LABEL_110;
            }
            v38 = 0;
            v31 = a2 - 1;
            v34 = *(_QWORD *)(v131 + 16);
LABEL_42:
            a6 = 1;
            a5 = 0;
            v39 = v31 & (v34 ^ (v34 >> 9));
            v40 = (__int64 *)(v32 + 32LL * v39);
            v6 = *v40;
            if ( *v40 == v34 )
            {
LABEL_43:
              v133 = (unsigned int *)v40[2];
              goto LABEL_44;
            }
            while ( v6 != -4 )
            {
              if ( !a5 && v6 == -16 )
                a5 = v40;
              v39 = v31 & (a6 + v39);
              v40 = (__int64 *)(v32 + 32LL * v39);
              v6 = *v40;
              if ( v34 == *v40 )
                goto LABEL_43;
              a6 = (unsigned int)(a6 + 1);
            }
            v74 = *(_DWORD *)(a1 + 40);
            if ( !a5 )
              a5 = v40;
            ++*(_QWORD *)(a1 + 24);
            v70 = v74 + 1;
            if ( 4 * (v74 + 1) < (unsigned int)(3 * a2) )
            {
              if ( (int)a2 - (v70 + *(_DWORD *)(a1 + 44)) > (unsigned int)a2 >> 3 )
                goto LABEL_112;
              v135 = v38;
              sub_D3A5D0(v125, a2);
              v75 = *(_DWORD *)(a1 + 48);
              if ( v75 )
              {
                a6 = (unsigned int)(v75 - 1);
                v76 = *(_QWORD *)(a1 + 32);
                a2 = 0;
                v77 = 1;
                v78 = *(_QWORD *)(v131 + 16);
                v70 = *(_DWORD *)(a1 + 40) + 1;
                v38 = v135;
                v79 = a6 & (v78 ^ (v78 >> 9));
                a5 = (__int64 *)(v76 + 32LL * v79);
                v80 = *a5;
                if ( *a5 != v78 )
                {
                  while ( v80 != -4 )
                  {
                    if ( v80 == -16 && !a2 )
                      a2 = (unsigned __int64)a5;
                    v79 = a6 & (v77 + v79);
                    a5 = (__int64 *)(v76 + 32LL * v79);
                    v80 = *a5;
                    if ( v78 == *a5 )
                      goto LABEL_112;
                    ++v77;
                  }
                  goto LABEL_129;
                }
                goto LABEL_112;
              }
LABEL_238:
              ++*(_DWORD *)(a1 + 40);
              BUG();
            }
LABEL_110:
            a2 = (unsigned int)(2 * a2);
            v134 = v38;
            sub_D3A5D0(v125, a2);
            v67 = *(_DWORD *)(a1 + 48);
            if ( !v67 )
              goto LABEL_238;
            a6 = (unsigned int)(v67 - 1);
            v68 = *(_QWORD *)(a1 + 32);
            v69 = *(_QWORD *)(v131 + 16);
            v70 = *(_DWORD *)(a1 + 40) + 1;
            v38 = v134;
            v71 = a6 & (v69 ^ (v69 >> 9));
            a5 = (__int64 *)(v68 + 32LL * v71);
            v72 = *a5;
            if ( v69 != *a5 )
            {
              v111 = 1;
              a2 = 0;
              while ( v72 != -4 )
              {
                if ( v72 == -16 && !a2 )
                  a2 = (unsigned __int64)a5;
                v71 = a6 & (v111 + v71);
                a5 = (__int64 *)(v68 + 32LL * v71);
                v72 = *a5;
                if ( v69 == *a5 )
                  goto LABEL_112;
                ++v111;
              }
LABEL_129:
              if ( a2 )
                a5 = (__int64 *)a2;
            }
LABEL_112:
            *(_DWORD *)(a1 + 40) = v70;
            if ( *a5 != -4 )
              --*(_DWORD *)(a1 + 44);
            v133 = 0;
            v73 = *(_QWORD *)(v131 + 16);
            a5[1] = 0;
            a5[2] = 0;
            *a5 = v73;
            a5[3] = 0;
          }
LABEL_44:
          v41 = v38;
          if ( v133 != v38 )
          {
            while ( 1 )
            {
              v44 = *v41;
              LODWORD(v6) = *v137;
              if ( *v41 >= *v137 )
              {
                v42 = (__int64 *)(v131 + 16);
                v43 = (__int64 *)(v130 + 16);
                v44 = *v137;
                LODWORD(v6) = *v41;
              }
              else
              {
                v42 = (__int64 *)(v130 + 16);
                v43 = (__int64 *)(v131 + 16);
              }
              v45 = sub_D3C740((__int64 *)a1, v43, v44, v42, v6);
              v46 = sub_D354B0(v45);
              a2 = v46;
              sub_D355F0(a1, v46);
              if ( !*(_BYTE *)(a1 + 232) )
                goto LABEL_51;
              v47 = *(_DWORD *)(a1 + 248);
              if ( !v45 )
              {
                if ( v47 < (unsigned int)qword_4F87228 )
                  goto LABEL_52;
LABEL_50:
                *(_BYTE *)(a1 + 232) = 0;
                *(_DWORD *)(a1 + 248) = 0;
                goto LABEL_51;
              }
              a2 = *(unsigned int *)(a1 + 252);
              v49 = v47;
              if ( v47 >= a2 )
              {
                if ( a2 < (unsigned __int64)v47 + 1 )
                {
                  a2 = a1 + 256;
                  sub_C8D5F0(a1 + 240, (const void *)(a1 + 256), v47 + 1LL, 0xCu, (__int64)a5, a6);
                  v49 = *(unsigned int *)(a1 + 248);
                }
                v53 = *(_QWORD *)(a1 + 240) + 12 * v49;
                *(_QWORD *)v53 = __PAIR64__(v6, v44);
                *(_DWORD *)(v53 + 8) = v45;
                v51 = *(_DWORD *)(a1 + 248) + 1;
                *(_DWORD *)(a1 + 248) = v51;
              }
              else
              {
                a2 = 3LL * v47;
                v50 = (unsigned int *)(*(_QWORD *)(a1 + 240) + 12LL * v47);
                if ( v50 )
                {
                  *v50 = v44;
                  v50[1] = v6;
                  v50[2] = v45;
                  v47 = *(_DWORD *)(a1 + 248);
                }
                v51 = v47 + 1;
                *(_DWORD *)(a1 + 248) = v51;
              }
              if ( (unsigned int)qword_4F87228 <= v51 )
                goto LABEL_50;
              if ( *(_BYTE *)(a1 + 232) )
              {
                if ( v133 == ++v41 )
                  break;
              }
              else
              {
LABEL_51:
                if ( *(_DWORD *)(a1 + 228) )
                {
                  LOBYTE(v10) = v143;
                  LODWORD(v6) = 0;
                  goto LABEL_76;
                }
LABEL_52:
                if ( v133 == ++v41 )
                  break;
              }
            }
          }
          if ( v129 != ++v137 )
            continue;
          break;
        }
LABEL_72:
        v131 = *(_QWORD *)(v131 + 8) & 0xFFFFFFFFFFFFFFFELL;
      }
      while ( v131 );
      v10 = v143;
      v130 = *(_QWORD *)(v130 + 8) & 0xFFFFFFFFFFFFFFFELL;
    }
    while ( v130 );
LABEL_8:
    ++v132;
  }
  while ( v127 != v132 );
  LOBYTE(v6) = *(_DWORD *)(a1 + 228) == 0;
LABEL_76:
  if ( !(_BYTE)v10 )
    _libc_free(v140, a2);
  return (unsigned int)v6;
}
