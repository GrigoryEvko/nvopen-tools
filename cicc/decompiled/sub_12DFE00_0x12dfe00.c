// Function: sub_12DFE00
// Address: 0x12dfe00
//
__int64 __fastcall sub_12DFE00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v4; // rbx
  int v5; // eax
  unsigned int v6; // edx
  __int64 v7; // r13
  _QWORD *v8; // r15
  _QWORD *v9; // r14
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r10
  unsigned int v14; // r9d
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  unsigned int v20; // r9d
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r10
  unsigned int v24; // eax
  _QWORD *v25; // r12
  __int64 v26; // rbx
  unsigned __int64 v27; // r14
  _QWORD *v28; // rax
  __int64 v29; // r15
  _QWORD *v30; // r13
  __int64 v31; // rax
  _BYTE *v32; // rsi
  _QWORD *v33; // rax
  unsigned int v34; // ecx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // r13
  __int64 v40; // rsi
  __int64 v41; // r10
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdx
  unsigned int v46; // r8d
  void (__fastcall *v47)(_QWORD, _QWORD, _QWORD); // r9
  __int64 *v48; // r14
  int v49; // esi
  int v50; // esi
  __int64 v51; // r10
  unsigned int v52; // edx
  int v53; // eax
  __int64 *v54; // rcx
  __int64 v55; // r8
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rdi
  int v60; // eax
  int v61; // esi
  int v62; // esi
  __int64 v63; // r10
  unsigned int v64; // edx
  __int64 v65; // r8
  __int64 *v66; // r11
  unsigned int v67; // esi
  __int64 v68; // r10
  unsigned int v69; // edx
  __int64 v70; // rax
  _QWORD *v71; // rdi
  int v72; // edx
  int v73; // edi
  int v74; // edi
  __int64 v75; // r10
  __int64 v76; // rsi
  int v77; // eax
  __int64 v78; // rdx
  _QWORD *v79; // r9
  __int64 v80; // rbx
  int v81; // ecx
  int v82; // ecx
  int v83; // eax
  int v84; // esi
  int v85; // esi
  __int64 v86; // r9
  __int64 v87; // r11
  __int64 v88; // rbx
  int v89; // ecx
  _QWORD *v90; // rdi
  int v91; // eax
  int v92; // ecx
  __int64 v93; // rbx
  __int64 v94; // rsi
  int v95; // edx
  _QWORD *v96; // r10
  __int64 v97; // r10
  int v98; // r11d
  int v99; // edx
  int v100; // eax
  int v101; // ecx
  unsigned int v102; // ecx
  __int64 v103; // rdi
  int v104; // r11d
  unsigned int v105; // ebx
  int v106; // ecx
  __int64 v107; // r11
  int v108; // r8d
  int v109; // ecx
  int v110; // eax
  int v111; // ecx
  __int64 v112; // r11
  __int64 v113; // rsi
  __int64 v114; // rbx
  int v115; // edi
  _QWORD *v116; // r9
  __int64 v117; // rcx
  __int64 v118; // rdi
  int v119; // r9d
  __int64 v120; // [rsp+20h] [rbp-90h]
  _BOOL4 v121; // [rsp+2Ch] [rbp-84h]
  __int64 v122; // [rsp+30h] [rbp-80h]
  __int64 v123; // [rsp+38h] [rbp-78h]
  __int64 *v124; // [rsp+38h] [rbp-78h]
  __int64 v125; // [rsp+40h] [rbp-70h]
  void (__fastcall *v126)(_QWORD, _QWORD, _QWORD); // [rsp+40h] [rbp-70h]
  int v127; // [rsp+40h] [rbp-70h]
  void (__fastcall *v128)(_QWORD, _QWORD, _QWORD); // [rsp+40h] [rbp-70h]
  int v129; // [rsp+40h] [rbp-70h]
  int v130; // [rsp+40h] [rbp-70h]
  int v131; // [rsp+48h] [rbp-68h]
  __int64 v132; // [rsp+48h] [rbp-68h]
  _QWORD *v133; // [rsp+58h] [rbp-58h] BYREF
  __int64 v134; // [rsp+60h] [rbp-50h] BYREF
  __int64 v135; // [rsp+68h] [rbp-48h]
  __int64 v136; // [rsp+70h] [rbp-40h]
  unsigned int v137; // [rsp+78h] [rbp-38h]

  v3 = a2;
  v4 = (_QWORD *)a1;
  v5 = *(_DWORD *)(a3 + 200);
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  if ( v5 )
  {
    v6 = *(_DWORD *)(a1 + 8);
    v121 = v5 > 1;
    if ( v6 )
    {
      v122 = a2;
      v7 = 0;
      v8 = (_QWORD *)a1;
      v125 = v6;
      v120 = a1 + 80;
      while ( 1 )
      {
        v131 = v7;
        v9 = *(_QWORD **)(*v8 + 8 * v7);
        v10 = (*(__int64 (__fastcall **)(_QWORD *))(*v9 + 112LL))(v9);
        if ( v10 )
          goto LABEL_4;
        v11 = sub_163A1D0(v9, a2);
        v12 = sub_163A340(v11, v9[2]);
        if ( !v12 || !*(_BYTE *)(v12 + 41) )
        {
          a2 = *((unsigned int *)v8 + 26);
          if ( !(_DWORD)a2 )
            goto LABEL_60;
          goto LABEL_9;
        }
        v67 = *((_DWORD *)v8 + 26);
        if ( !v67 )
          break;
        v68 = v8[11];
        v69 = (v67 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v70 = v68 + 16LL * v69;
        v71 = *(_QWORD **)v70;
        if ( v9 == *(_QWORD **)v70 )
        {
          v72 = *(_DWORD *)(v70 + 8) | 2;
          goto LABEL_59;
        }
        v107 = 0;
        v108 = 1;
        while ( v71 != (_QWORD *)-8LL )
        {
          if ( v107 || v71 != (_QWORD *)-16LL )
            v70 = v107;
          v69 = (v67 - 1) & (v108 + v69);
          v117 = v68 + 16LL * v69;
          v71 = *(_QWORD **)v117;
          if ( v9 == *(_QWORD **)v117 )
          {
            v70 = v68 + 16LL * v69;
            v72 = *(_DWORD *)(v117 + 8) | 2;
            goto LABEL_59;
          }
          ++v108;
          v107 = v70;
          v70 = v68 + 16LL * v69;
        }
        v109 = *((_DWORD *)v8 + 24);
        if ( v107 )
          v70 = v107;
        ++v8[10];
        v95 = v109 + 1;
        if ( 4 * (v109 + 1) >= 3 * v67 )
          goto LABEL_90;
        if ( v67 - *((_DWORD *)v8 + 25) - v95 <= v67 >> 3 )
        {
          sub_12DDEF0(v120, v67);
          v110 = *((_DWORD *)v8 + 26);
          if ( !v110 )
          {
LABEL_198:
            ++*((_DWORD *)v8 + 24);
            BUG();
          }
          v111 = v110 - 1;
          v112 = v8[11];
          v113 = 0;
          LODWORD(v114) = (v110 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v95 = *((_DWORD *)v8 + 24) + 1;
          v115 = 1;
          v70 = v112 + 16LL * (unsigned int)v114;
          v116 = *(_QWORD **)v70;
          if ( v9 != *(_QWORD **)v70 )
          {
            while ( v116 != (_QWORD *)-8LL )
            {
              if ( !v113 && v116 == (_QWORD *)-16LL )
                v113 = v70;
              v114 = v111 & (unsigned int)(v114 + v115);
              v70 = v112 + 16 * v114;
              v116 = *(_QWORD **)v70;
              if ( v9 == *(_QWORD **)v70 )
                goto LABEL_92;
              ++v115;
            }
            if ( v113 )
              v70 = v113;
          }
        }
LABEL_92:
        *((_DWORD *)v8 + 24) = v95;
        if ( *(_QWORD *)v70 != -8 )
          --*((_DWORD *)v8 + 25);
        *(_QWORD *)v70 = v9;
        v72 = 2;
        *(_DWORD *)(v70 + 8) = 0;
LABEL_59:
        *(_DWORD *)(v70 + 8) = v72;
        a2 = *((unsigned int *)v8 + 26);
        if ( !(_DWORD)a2 )
        {
LABEL_60:
          ++v8[10];
          goto LABEL_61;
        }
LABEL_9:
        v13 = v8[11];
        v14 = (a2 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v15 = v13 + 16LL * v14;
        v16 = *(_QWORD **)v15;
        if ( v9 == *(_QWORD **)v15 )
        {
LABEL_10:
          if ( (*(_BYTE *)(v15 + 8) & 2) == 0 )
            goto LABEL_11;
LABEL_4:
          if ( v125 == ++v7 )
            goto LABEL_28;
        }
        else
        {
          v78 = 0;
          v82 = 1;
          while ( v16 != (_QWORD *)-8LL )
          {
            if ( !v78 && v16 == (_QWORD *)-16LL )
              v78 = v15;
            v14 = (a2 - 1) & (v82 + v14);
            v15 = v13 + 16LL * v14;
            v16 = *(_QWORD **)v15;
            if ( v9 == *(_QWORD **)v15 )
              goto LABEL_10;
            ++v82;
          }
          if ( !v78 )
            v78 = v15;
          v83 = *((_DWORD *)v8 + 24);
          ++v8[10];
          v77 = v83 + 1;
          if ( 4 * v77 < (unsigned int)(3 * a2) )
          {
            if ( (int)a2 - *((_DWORD *)v8 + 25) - v77 <= (unsigned int)a2 >> 3 )
            {
              sub_12DDEF0(v120, a2);
              v84 = *((_DWORD *)v8 + 26);
              if ( !v84 )
                goto LABEL_198;
              v85 = v84 - 1;
              v86 = v8[11];
              v87 = 0;
              LODWORD(v88) = v85 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v89 = 1;
              v77 = *((_DWORD *)v8 + 24) + 1;
              v78 = v86 + 16LL * (unsigned int)v88;
              v90 = *(_QWORD **)v78;
              if ( v9 != *(_QWORD **)v78 )
              {
                while ( v90 != (_QWORD *)-8LL )
                {
                  if ( !v87 && v90 == (_QWORD *)-16LL )
                    v87 = v78;
                  v88 = v85 & (unsigned int)(v88 + v89);
                  v78 = v86 + 16 * v88;
                  v90 = *(_QWORD **)v78;
                  if ( v9 == *(_QWORD **)v78 )
                    goto LABEL_79;
                  ++v89;
                }
                if ( v87 )
                  v78 = v87;
              }
            }
            goto LABEL_79;
          }
LABEL_61:
          sub_12DDEF0(v120, 2 * a2);
          v73 = *((_DWORD *)v8 + 26);
          if ( !v73 )
            goto LABEL_198;
          v74 = v73 - 1;
          v75 = v8[11];
          LODWORD(v76) = v74 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v77 = *((_DWORD *)v8 + 24) + 1;
          v78 = v75 + 16LL * (unsigned int)v76;
          v79 = *(_QWORD **)v78;
          if ( v9 != *(_QWORD **)v78 )
          {
            v80 = 0;
            v81 = 1;
            while ( v79 != (_QWORD *)-8LL )
            {
              if ( !v80 && v79 == (_QWORD *)-16LL )
                v80 = v78;
              v76 = v74 & (unsigned int)(v76 + v81);
              v78 = v75 + 16 * v76;
              v79 = *(_QWORD **)v78;
              if ( v9 == *(_QWORD **)v78 )
                goto LABEL_79;
              ++v81;
            }
            if ( v80 )
              v78 = v80;
          }
LABEL_79:
          *((_DWORD *)v8 + 24) = v77;
          if ( *(_QWORD *)v78 != -8 )
            --*((_DWORD *)v8 + 25);
          *(_QWORD *)v78 = v9;
          *(_DWORD *)(v78 + 8) = 0;
LABEL_11:
          v17 = (*(__int64 (__fastcall **)(_QWORD *))(*v9 + 16LL))(v9);
          a2 = v137;
          v18 = v135;
          v123 = v17;
          v19 = v17;
          if ( !v137 )
            goto LABEL_109;
          v20 = v137 - 1;
          v21 = (v137 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v22 = (__int64 *)(v135 + 16LL * v21);
          v23 = *v22;
          if ( v19 == *v22 )
          {
LABEL_13:
            if ( v22 != (__int64 *)(v135 + 16LL * v137) )
            {
              v24 = *((_DWORD *)v22 + 2) + v121;
              if ( v24 < (unsigned int)v7 )
              {
                v25 = v9;
                v26 = 8LL * v24;
                v27 = 8 * (v24 + (unsigned __int64)((_DWORD)v7 - 1 - v24) + 1);
                v28 = v8;
                v29 = v7;
                v30 = v28;
                while ( 1 )
                {
                  if ( (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v30 + v26) + 112LL))(*(_QWORD *)(*v30 + v26)) )
                    goto LABEL_16;
                  v31 = *(_QWORD *)(*v30 + v26);
                  if ( !v31 )
                    goto LABEL_16;
                  *((_BYTE *)v25 + 152) = 1;
                  v32 = *(_BYTE **)(v31 + 40);
                  v133 = v25;
                  if ( v32 == *(_BYTE **)(v31 + 48) )
                  {
                    sub_12DCE20(v31 + 32, v32, &v133);
LABEL_16:
                    v26 += 8;
                    if ( v27 == v26 )
                      goto LABEL_23;
                  }
                  else
                  {
                    if ( v32 )
                    {
                      *(_QWORD *)v32 = v25;
                      v32 = *(_BYTE **)(v31 + 40);
                    }
                    v26 += 8;
                    *(_QWORD *)(v31 + 40) = v32 + 8;
                    if ( v27 == v26 )
                    {
LABEL_23:
                      v33 = v30;
                      v10 = 0;
                      a2 = v137;
                      v7 = v29;
                      v18 = v135;
                      v8 = v33;
                      goto LABEL_24;
                    }
                  }
                }
              }
            }
          }
          else
          {
            v100 = 1;
            while ( v23 != -1 )
            {
              v101 = v100 + 1;
              v21 = v20 & (v100 + v21);
              v22 = (__int64 *)(v135 + 16LL * v21);
              v23 = *v22;
              if ( v123 == *v22 )
                goto LABEL_13;
              v100 = v101;
            }
LABEL_24:
            if ( !(_DWORD)a2 )
            {
LABEL_109:
              ++v134;
              goto LABEL_110;
            }
            v20 = a2 - 1;
          }
          v34 = v20 & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
          v35 = v18 + 16LL * v34;
          v36 = *(_QWORD *)v35;
          if ( v123 == *(_QWORD *)v35 )
            goto LABEL_27;
          v97 = 0;
          v98 = 1;
          while ( v36 != -1 )
          {
            if ( v97 || v36 != -2 )
              v35 = v97;
            v34 = v20 & (v98 + v34);
            v36 = *(_QWORD *)(v18 + 16LL * v34);
            if ( v123 == v36 )
            {
              v35 = v18 + 16LL * v34;
              goto LABEL_27;
            }
            ++v98;
            v97 = v35;
            v35 = v18 + 16LL * v34;
          }
          if ( v97 )
            v35 = v97;
          ++v134;
          v99 = v136 + 1;
          if ( 4 * ((int)v136 + 1) < (unsigned int)(3 * a2) )
          {
            if ( (int)a2 - (v99 + HIDWORD(v136)) > (unsigned int)a2 >> 3 )
              goto LABEL_102;
            sub_12DFC30((__int64)&v134, a2);
            if ( !v137 )
            {
LABEL_200:
              LODWORD(v136) = v136 + 1;
              BUG();
            }
            v105 = (v137 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
            v99 = v136 + 1;
            v106 = 1;
            v35 = v135 + 16LL * v105;
            a2 = *(_QWORD *)v35;
            if ( v123 == *(_QWORD *)v35 )
              goto LABEL_102;
            while ( a2 != -1 )
            {
              if ( !v10 && a2 == -2 )
                v10 = v35;
              v105 = (v137 - 1) & (v106 + v105);
              v35 = v135 + 16LL * v105;
              a2 = *(_QWORD *)v35;
              if ( v123 == *(_QWORD *)v35 )
                goto LABEL_102;
              ++v106;
            }
            goto LABEL_114;
          }
LABEL_110:
          sub_12DFC30((__int64)&v134, 2 * a2);
          if ( !v137 )
            goto LABEL_200;
          a2 = v137 - 1;
          v99 = v136 + 1;
          v102 = a2 & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
          v35 = v135 + 16LL * v102;
          v103 = *(_QWORD *)v35;
          if ( v123 == *(_QWORD *)v35 )
            goto LABEL_102;
          v104 = 1;
          while ( v103 != -1 )
          {
            if ( !v10 && v103 == -2 )
              v10 = v35;
            v102 = a2 & (v104 + v102);
            v35 = v135 + 16LL * v102;
            v103 = *(_QWORD *)v35;
            if ( v123 == *(_QWORD *)v35 )
              goto LABEL_102;
            ++v104;
          }
LABEL_114:
          if ( v10 )
            v35 = v10;
LABEL_102:
          LODWORD(v136) = v99;
          if ( *(_QWORD *)v35 != -1 )
            --HIDWORD(v136);
          *(_DWORD *)(v35 + 8) = 0;
          *(_QWORD *)v35 = v123;
LABEL_27:
          ++v7;
          *(_DWORD *)(v35 + 8) = v131;
          if ( v125 == v7 )
          {
LABEL_28:
            v4 = v8;
            v3 = v122;
            goto LABEL_29;
          }
        }
      }
      ++v8[10];
LABEL_90:
      sub_12DDEF0(v120, 2 * v67);
      v91 = *((_DWORD *)v8 + 26);
      if ( !v91 )
        goto LABEL_198;
      v92 = v91 - 1;
      v93 = v8[11];
      LODWORD(v94) = (v91 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v95 = *((_DWORD *)v8 + 24) + 1;
      v70 = v93 + 16LL * (unsigned int)v94;
      v96 = *(_QWORD **)v70;
      if ( v9 != *(_QWORD **)v70 )
      {
        v118 = 0;
        v119 = 1;
        while ( v96 != (_QWORD *)-8LL )
        {
          if ( !v118 && v96 == (_QWORD *)-16LL )
            v118 = v70;
          v94 = v92 & (unsigned int)(v94 + v119);
          v70 = v93 + 16 * v94;
          v96 = *(_QWORD **)v70;
          if ( v9 == *(_QWORD **)v70 )
            goto LABEL_92;
          ++v119;
        }
        if ( v118 )
          v70 = v118;
      }
      goto LABEL_92;
    }
    v58 = 0;
  }
  else
  {
LABEL_29:
    v37 = *((unsigned int *)v4 + 2);
    if ( (_DWORD)v37 )
    {
      v38 = 0;
      v132 = (__int64)(v4 + 10);
      v39 = 8 * v37;
      do
      {
        while ( 1 )
        {
          v46 = *((_DWORD *)v4 + 26);
          v47 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(*(_QWORD *)v3 + 16LL);
          v48 = (__int64 *)(v38 + *v4);
          if ( !v46 )
            break;
          v40 = *v48;
          v41 = v4[11];
          v42 = (v46 - 1) & (((unsigned int)*v48 >> 9) ^ ((unsigned int)*v48 >> 4));
          v43 = (__int64 *)(v41 + 16LL * v42);
          v44 = *v43;
          if ( *v43 == *v48 )
          {
            v45 = v43[1] & 1;
            goto LABEL_33;
          }
          v127 = 1;
          v54 = 0;
          while ( 1 )
          {
            if ( v44 == -8 )
            {
              if ( !v54 )
                v54 = v43;
              v60 = *((_DWORD *)v4 + 24);
              ++v4[10];
              v53 = v60 + 1;
              if ( 4 * v53 >= 3 * v46 )
                goto LABEL_36;
              if ( v46 - *((_DWORD *)v4 + 25) - v53 <= v46 >> 3 )
              {
                v128 = v47;
                sub_12DDEF0(v132, v46);
                v61 = *((_DWORD *)v4 + 26);
                if ( !v61 )
                {
LABEL_199:
                  ++*((_DWORD *)v4 + 24);
                  BUG();
                }
                v62 = v61 - 1;
                v63 = v4[11];
                v47 = v128;
                v64 = v62 & (((unsigned int)*v48 >> 9) ^ ((unsigned int)*v48 >> 4));
                v53 = *((_DWORD *)v4 + 24) + 1;
                v54 = (__int64 *)(v63 + 16LL * v64);
                v65 = *v54;
                if ( *v54 != *v48 )
                {
                  v129 = 1;
                  v66 = 0;
                  while ( v65 != -8 )
                  {
                    if ( v65 != -16 || v66 )
                      v54 = v66;
                    v64 = v62 & (v129 + v64);
                    v65 = *(_QWORD *)(v63 + 16LL * v64);
                    if ( *v48 == v65 )
                    {
                      v54 = (__int64 *)(v63 + 16LL * v64);
                      goto LABEL_38;
                    }
                    ++v129;
                    v66 = v54;
                    v54 = (__int64 *)(v63 + 16LL * v64);
                  }
                  goto LABEL_53;
                }
              }
              goto LABEL_38;
            }
            if ( v44 != -16 || v54 )
              v43 = v54;
            v42 = (v46 - 1) & (v127 + v42);
            v124 = (__int64 *)(v41 + 16LL * v42);
            v44 = *v124;
            if ( v40 == *v124 )
              break;
            ++v127;
            v54 = v43;
            v43 = (__int64 *)(v41 + 16LL * v42);
          }
          v45 = v124[1] & 1;
LABEL_33:
          v38 += 8;
          v47(v3, v40, v45);
          if ( v39 == v38 )
            goto LABEL_41;
        }
        ++v4[10];
LABEL_36:
        v126 = v47;
        sub_12DDEF0(v132, 2 * v46);
        v49 = *((_DWORD *)v4 + 26);
        if ( !v49 )
          goto LABEL_199;
        v50 = v49 - 1;
        v51 = v4[11];
        v47 = v126;
        v52 = v50 & (((unsigned int)*v48 >> 9) ^ ((unsigned int)*v48 >> 4));
        v53 = *((_DWORD *)v4 + 24) + 1;
        v54 = (__int64 *)(v51 + 16LL * v52);
        v55 = *v54;
        if ( *v48 != *v54 )
        {
          v130 = 1;
          v66 = 0;
          while ( v55 != -8 )
          {
            if ( v55 == -16 && !v66 )
              v66 = v54;
            v52 = v50 & (v130 + v52);
            v54 = (__int64 *)(v51 + 16LL * v52);
            v55 = *v54;
            if ( *v48 == *v54 )
              goto LABEL_38;
            ++v130;
          }
LABEL_53:
          if ( v66 )
            v54 = v66;
        }
LABEL_38:
        *((_DWORD *)v4 + 24) = v53;
        if ( *v54 != -8 )
          --*((_DWORD *)v4 + 25);
        v56 = *v48;
        *((_DWORD *)v54 + 2) = 0;
        *v54 = v56;
        v57 = *(_QWORD *)(*v4 + v38);
        v38 += 8;
        v47(v3, v57, 0);
      }
      while ( v39 != v38 );
    }
LABEL_41:
    v58 = v135;
  }
  return j___libc_free_0(v58);
}
