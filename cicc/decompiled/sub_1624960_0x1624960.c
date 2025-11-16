// Function: sub_1624960
// Address: 0x1624960
//
void __fastcall sub_1624960(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v4; // r14
  __int64 v5; // rax
  unsigned int *v6; // rcx
  unsigned int *v7; // rax
  int *v8; // r13
  unsigned int v9; // ecx
  unsigned int v10; // edx
  int *v11; // rax
  _BOOL4 v12; // r15d
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r13
  __int64 v16; // rdi
  unsigned int v17; // edx
  __int64 v18; // rcx
  __int64 *v19; // rbx
  __int64 v20; // rcx
  char *v21; // r13
  __int64 v22; // rcx
  char *v23; // rbx
  unsigned int *v24; // rdx
  char *v25; // r10
  char *v26; // r9
  char *v27; // r8
  unsigned int *v28; // rax
  unsigned int *v29; // rax
  unsigned int *v30; // rax
  unsigned int *v31; // rax
  unsigned int *v32; // rax
  unsigned int *v33; // rcx
  unsigned int *v34; // rax
  unsigned int *v35; // rdx
  int *v36; // r15
  unsigned int *v37; // rbx
  unsigned int v38; // ecx
  int *i; // r13
  unsigned int v40; // edx
  int *v41; // rax
  _BOOL4 v42; // r15d
  __int64 v43; // rax
  int v44; // edx
  unsigned int v45; // ecx
  unsigned int v46; // edx
  int *v47; // rax
  _BOOL4 v48; // r13d
  __int64 v49; // rax
  int v50; // eax
  int v51; // ecx
  __int64 v52; // rdi
  unsigned int v53; // eax
  int v54; // esi
  __int64 *v55; // rbx
  __int64 v56; // rdx
  __int64 v57; // r13
  unsigned __int64 v58; // r12
  __int64 v59; // rsi
  unsigned int v60; // r14d
  __int64 v61; // rax
  int *v62; // r15
  __int64 v63; // r11
  __int64 v64; // rcx
  unsigned int v65; // r14d
  __int64 v66; // rax
  int *v67; // r15
  __int64 v68; // r11
  __int64 v69; // rcx
  unsigned int v70; // r14d
  __int64 v71; // rax
  int *v72; // r15
  __int64 v73; // r11
  __int64 v74; // rcx
  unsigned int v75; // r14d
  __int64 v76; // rax
  int *v77; // r15
  __int64 v78; // r11
  __int64 v79; // rcx
  char *v80; // r14
  unsigned int *v81; // rax
  unsigned int *v82; // rcx
  unsigned int v83; // edx
  __int64 v84; // rsi
  unsigned __int8 *v85; // rsi
  char *v86; // rax
  __int64 v87; // rax
  int *v88; // rdi
  __int64 v89; // rsi
  __int64 v90; // rcx
  __int64 v91; // r12
  __int64 v92; // r15
  __int64 *v93; // r13
  __int64 *v94; // r14
  unsigned __int8 *v95; // rsi
  __int64 v96; // rsi
  __int64 v97; // rbx
  int v98; // eax
  int v99; // ecx
  __int64 v100; // rdi
  unsigned int v101; // eax
  int v102; // esi
  __int64 v103; // rdx
  __int64 v104; // r13
  __int64 v105; // rsi
  __int64 v106; // rcx
  int v107; // eax
  unsigned int *v108; // rax
  unsigned int *v109; // rcx
  __int64 *v110; // r11
  int v111; // r10d
  __int64 *v112; // r9
  __int64 v113; // rax
  unsigned int v114; // edx
  int *v115; // rdi
  __int64 v116; // rax
  unsigned int v117; // ecx
  int *v118; // rsi
  __int64 v119; // rax
  unsigned int v120; // esi
  int *v121; // rcx
  __int64 v122; // [rsp+10h] [rbp-C0h]
  __int64 *v123; // [rsp+18h] [rbp-B8h]
  char *v124; // [rsp+20h] [rbp-B0h]
  __int64 v126; // [rsp+30h] [rbp-A0h]
  unsigned int *v127; // [rsp+38h] [rbp-98h]
  __int64 v128; // [rsp+40h] [rbp-90h] BYREF
  __int64 *v129; // [rsp+48h] [rbp-88h] BYREF
  unsigned int *v130; // [rsp+50h] [rbp-80h] BYREF
  __int64 v131; // [rsp+58h] [rbp-78h]
  _BYTE v132[24]; // [rsp+60h] [rbp-70h] BYREF
  int v133; // [rsp+78h] [rbp-58h] BYREF
  int *v134; // [rsp+80h] [rbp-50h]
  int *v135; // [rsp+88h] [rbp-48h]
  int *v136; // [rsp+90h] [rbp-40h]
  __int64 v137; // [rsp+98h] [rbp-38h]

  if ( *(__int16 *)(a1 + 18) < 0 )
  {
    v4 = a2;
    v5 = *(_QWORD *)sub_16498A0(a1);
    v133 = 0;
    v134 = 0;
    v122 = v5;
    v130 = (unsigned int *)v132;
    v131 = 0x400000000LL;
    v135 = &v133;
    v136 = &v133;
    v137 = 0;
    v127 = &a2[a3];
    if ( a2 == v127 )
      goto LABEL_100;
LABEL_4:
    v6 = &v130[(unsigned int)v131];
    if ( v130 != v6 )
    {
      v7 = v130;
      while ( *v7 != *v4 )
      {
        if ( v6 == ++v7 )
          goto LABEL_62;
      }
      if ( v6 != v7 )
        goto LABEL_9;
    }
LABEL_62:
    v36 = v134;
    if ( (unsigned int)v131 <= 3uLL )
    {
      if ( (unsigned int)v131 >= HIDWORD(v131) )
      {
        sub_16CD150(&v130, v132, 0, 4);
        v6 = &v130[(unsigned int)v131];
      }
      *v6 = *v4;
      LODWORD(v131) = v131 + 1;
    }
    else
    {
      v37 = &v130[(unsigned int)v131 - 1];
      if ( v134 )
        goto LABEL_64;
LABEL_76:
      i = &v133;
      if ( v135 == &v133 )
      {
        v42 = 1;
      }
      else
      {
        while ( *(_DWORD *)(sub_220EF80(i) + 32) >= *v37 )
        {
          v44 = v131 - 1;
          LODWORD(v131) = v44;
          if ( !v44 )
            goto LABEL_81;
LABEL_75:
          v37 = &v130[v44 - 1];
          if ( !v36 )
            goto LABEL_76;
LABEL_64:
          v38 = *v37;
          for ( i = v36; ; i = v41 )
          {
            v40 = i[8];
            v41 = (int *)*((_QWORD *)i + 3);
            if ( v38 < v40 )
              v41 = (int *)*((_QWORD *)i + 2);
            if ( !v41 )
              break;
          }
          if ( v38 >= v40 )
          {
            if ( v38 <= v40 )
              goto LABEL_74;
            break;
          }
          if ( v135 == i )
            break;
        }
        v42 = 1;
        if ( i != &v133 )
          v42 = *v37 < i[8];
      }
      v43 = sub_22077B0(40);
      *(_DWORD *)(v43 + 32) = *v37;
      sub_220F040(v42, v43, i, &v133);
      ++v137;
      v36 = v134;
LABEL_74:
      v44 = v131 - 1;
      LODWORD(v131) = v44;
      if ( v44 )
        goto LABEL_75;
LABEL_81:
      if ( v36 )
      {
        v45 = *v4;
        while ( 1 )
        {
          v46 = v36[8];
          v47 = (int *)*((_QWORD *)v36 + 3);
          if ( v45 < v46 )
            v47 = (int *)*((_QWORD *)v36 + 2);
          if ( !v47 )
            break;
          v36 = v47;
        }
        if ( v45 >= v46 )
        {
          if ( v45 <= v46 )
            goto LABEL_9;
          goto LABEL_89;
        }
        if ( v135 == v36 )
        {
LABEL_89:
          v48 = 1;
          if ( v36 != &v133 )
            v48 = *v4 < v36[8];
          goto LABEL_91;
        }
      }
      else
      {
        v36 = &v133;
        if ( v135 == &v133 )
        {
          v48 = 1;
LABEL_91:
          v49 = sub_22077B0(40);
          *(_DWORD *)(v49 + 32) = *v4;
          sub_220F040(v48, v49, v36, &v133);
          ++v137;
          goto LABEL_9;
        }
      }
      if ( *(_DWORD *)(sub_220EF80(v36) + 32) < *v4 )
        goto LABEL_89;
    }
LABEL_9:
    for ( ++v4; v4 != v127; ++v137 )
    {
      if ( !v137 )
        goto LABEL_4;
      v8 = v134;
      if ( v134 )
      {
        v9 = *v4;
        while ( 1 )
        {
          v10 = v8[8];
          v11 = (int *)*((_QWORD *)v8 + 3);
          if ( v9 < v10 )
            v11 = (int *)*((_QWORD *)v8 + 2);
          if ( !v11 )
            break;
          v8 = v11;
        }
        if ( v9 >= v10 )
        {
          if ( v9 <= v10 )
            goto LABEL_9;
LABEL_19:
          v12 = 1;
          if ( v8 != &v133 )
            goto LABEL_95;
          goto LABEL_20;
        }
        if ( v135 == v8 )
          goto LABEL_19;
      }
      else
      {
        v8 = &v133;
        if ( v135 == &v133 )
        {
          v12 = 1;
          goto LABEL_20;
        }
      }
      if ( *(_DWORD *)(sub_220EF80(v8) + 32) >= *v4 )
        goto LABEL_9;
      v12 = 1;
      if ( v8 != &v133 )
LABEL_95:
        v12 = *v4 < v8[8];
LABEL_20:
      ++v4;
      v13 = sub_22077B0(40);
      *(_DWORD *)(v13 + 32) = *(v4 - 1);
      sub_220F040(v12, v13, v8, &v133);
    }
    if ( !(_DWORD)v131 && !v137 )
    {
LABEL_100:
      v50 = *(_DWORD *)(v122 + 2728);
      if ( v50 )
      {
        v51 = v50 - 1;
        v52 = *(_QWORD *)(v122 + 2712);
        v53 = (v50 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v54 = 1;
        v55 = (__int64 *)(v52 + 56LL * v53);
        v56 = *v55;
        if ( a1 != *v55 )
        {
          while ( v56 != -8 )
          {
            v53 = v51 & (v54 + v53);
            v55 = (__int64 *)(v52 + 56LL * v53);
            v56 = *v55;
            if ( a1 == *v55 )
              goto LABEL_102;
            ++v54;
          }
          goto LABEL_110;
        }
LABEL_102:
        v57 = v55[1];
        v58 = v57 + 16LL * *((unsigned int *)v55 + 4);
        if ( v57 == v58 )
          goto LABEL_107;
        do
        {
          v59 = *(_QWORD *)(v58 - 8);
          v58 -= 16LL;
          if ( v59 )
            sub_161E7C0(v58 + 8, v59);
        }
        while ( v57 != v58 );
        goto LABEL_106;
      }
      goto LABEL_110;
    }
    v14 = *(_DWORD *)(v122 + 2728);
    v128 = a1;
    v15 = v122 + 2704;
    if ( v14 )
    {
      v16 = *(_QWORD *)(v122 + 2712);
      v17 = (v14 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v18 = *(_QWORD *)(v16 + 56LL * v17);
      v123 = (__int64 *)(v16 + 56LL * v17);
      if ( a1 == v18 )
      {
        v19 = (__int64 *)(v16 + 56LL * v17);
LABEL_25:
        v124 = (char *)v19[1];
        v20 = 16LL * *((unsigned int *)v19 + 4);
        v21 = &v124[v20];
        v126 = v20 >> 6;
        v22 = v20 >> 4;
        if ( v126 )
        {
          v23 = (char *)v19[1];
          v24 = &v130[(unsigned int)v131];
          v25 = v124 + 48;
          v26 = v124 + 32;
          v27 = v124 + 16;
          while ( 1 )
          {
            if ( v137 )
            {
              if ( !v134 )
                goto LABEL_140;
              v60 = *(_DWORD *)v23;
              v61 = (__int64)v134;
              v62 = &v133;
              do
              {
                while ( 1 )
                {
                  v63 = *(_QWORD *)(v61 + 16);
                  v64 = *(_QWORD *)(v61 + 24);
                  if ( *(_DWORD *)(v61 + 32) >= v60 )
                    break;
                  v61 = *(_QWORD *)(v61 + 24);
                  if ( !v64 )
                    goto LABEL_118;
                }
                v62 = (int *)v61;
                v61 = *(_QWORD *)(v61 + 16);
              }
              while ( v63 );
LABEL_118:
              if ( v62 == &v133 || v60 < v62[8] )
                goto LABEL_140;
              v65 = *((_DWORD *)v23 + 4);
              v66 = (__int64)v134;
              v67 = &v133;
              do
              {
                while ( 1 )
                {
                  v68 = *(_QWORD *)(v66 + 16);
                  v69 = *(_QWORD *)(v66 + 24);
                  if ( *(_DWORD *)(v66 + 32) >= v65 )
                    break;
                  v66 = *(_QWORD *)(v66 + 24);
                  if ( !v69 )
                    goto LABEL_124;
                }
                v67 = (int *)v66;
                v66 = *(_QWORD *)(v66 + 16);
              }
              while ( v68 );
LABEL_124:
              if ( v67 == &v133 || v65 < v67[8] )
              {
LABEL_139:
                v23 = v27;
                goto LABEL_140;
              }
              v70 = *((_DWORD *)v23 + 8);
              v71 = (__int64)v134;
              v72 = &v133;
              do
              {
                while ( 1 )
                {
                  v73 = *(_QWORD *)(v71 + 16);
                  v74 = *(_QWORD *)(v71 + 24);
                  if ( *(_DWORD *)(v71 + 32) >= v70 )
                    break;
                  v71 = *(_QWORD *)(v71 + 24);
                  if ( !v74 )
                    goto LABEL_130;
                }
                v72 = (int *)v71;
                v71 = *(_QWORD *)(v71 + 16);
              }
              while ( v73 );
LABEL_130:
              if ( v72 == &v133 || v70 < v72[8] )
              {
LABEL_186:
                v23 = v26;
                goto LABEL_140;
              }
              v75 = *((_DWORD *)v23 + 12);
              v76 = (__int64)v134;
              v77 = &v133;
              do
              {
                while ( 1 )
                {
                  v78 = *(_QWORD *)(v76 + 16);
                  v79 = *(_QWORD *)(v76 + 24);
                  if ( *(_DWORD *)(v76 + 32) >= v75 )
                    break;
                  v76 = *(_QWORD *)(v76 + 24);
                  if ( !v79 )
                    goto LABEL_136;
                }
                v77 = (int *)v76;
                v76 = *(_QWORD *)(v76 + 16);
              }
              while ( v78 );
LABEL_136:
              if ( v77 == &v133 || v75 < v77[8] )
              {
LABEL_138:
                v23 = v25;
                goto LABEL_140;
              }
            }
            else
            {
              if ( v130 == v24 )
                goto LABEL_140;
              v28 = v130;
              while ( *v28 != *(_DWORD *)v23 )
              {
                if ( v24 == ++v28 )
                  goto LABEL_140;
              }
              if ( v24 == v28 )
                goto LABEL_140;
              v29 = v130;
              while ( *v29 != *((_DWORD *)v23 + 4) )
              {
                if ( v24 == ++v29 )
                  goto LABEL_139;
              }
              if ( v24 == v29 )
                goto LABEL_139;
              v30 = v130;
              while ( *v30 != *((_DWORD *)v23 + 8) )
              {
                if ( v24 == ++v30 )
                  goto LABEL_186;
              }
              if ( v24 == v30 )
                goto LABEL_186;
              v31 = v130;
              while ( *v31 != *((_DWORD *)v23 + 12) )
              {
                if ( v24 == ++v31 )
                  goto LABEL_138;
              }
              if ( v24 == v31 )
                goto LABEL_138;
            }
            v23 += 64;
            v25 += 64;
            v26 += 64;
            v27 += 64;
            if ( !--v126 )
            {
              v22 = (v21 - v23) >> 4;
              goto LABEL_47;
            }
          }
        }
        v23 = (char *)v19[1];
LABEL_47:
        if ( v22 != 2 )
        {
          if ( v22 != 3 )
          {
            if ( v22 != 1 )
              goto LABEL_61;
            goto LABEL_55;
          }
          if ( v137 )
          {
            v119 = (__int64)v134;
            if ( !v134 )
              goto LABEL_140;
            v120 = *(_DWORD *)v23;
            v121 = &v133;
            do
            {
              if ( *(_DWORD *)(v119 + 32) < v120 )
              {
                v119 = *(_QWORD *)(v119 + 24);
              }
              else
              {
                v121 = (int *)v119;
                v119 = *(_QWORD *)(v119 + 16);
              }
            }
            while ( v119 );
            if ( v121 == &v133 || v120 < v121[8] )
              goto LABEL_140;
          }
          else
          {
            v108 = v130;
            v109 = &v130[(unsigned int)v131];
            if ( v130 == v109 )
              goto LABEL_140;
            while ( *v108 != *(_DWORD *)v23 )
            {
              if ( v109 == ++v108 )
                goto LABEL_140;
            }
            if ( v109 == v108 )
              goto LABEL_140;
          }
          v23 += 16;
        }
        if ( v137 )
        {
          v116 = (__int64)v134;
          if ( !v134 )
            goto LABEL_140;
          v117 = *(_DWORD *)v23;
          v118 = &v133;
          do
          {
            if ( *(_DWORD *)(v116 + 32) < v117 )
            {
              v116 = *(_QWORD *)(v116 + 24);
            }
            else
            {
              v118 = (int *)v116;
              v116 = *(_QWORD *)(v116 + 16);
            }
          }
          while ( v116 );
          if ( v118 == &v133 || v117 < v118[8] )
            goto LABEL_140;
        }
        else
        {
          v32 = v130;
          v33 = &v130[(unsigned int)v131];
          if ( v130 == v33 )
            goto LABEL_140;
          while ( *v32 != *(_DWORD *)v23 )
          {
            if ( v33 == ++v32 )
              goto LABEL_140;
          }
          if ( v33 == v32 )
            goto LABEL_140;
        }
        v23 += 16;
LABEL_55:
        if ( !v137 )
        {
          v34 = v130;
          v35 = &v130[(unsigned int)v131];
          if ( v130 != v35 )
          {
            while ( *v34 != *(_DWORD *)v23 )
            {
              if ( v35 == ++v34 )
                goto LABEL_140;
            }
            if ( v35 != v34 )
              goto LABEL_61;
          }
          goto LABEL_140;
        }
        v113 = (__int64)v134;
        if ( !v134 )
          goto LABEL_140;
        v114 = *(_DWORD *)v23;
        v115 = &v133;
        do
        {
          if ( *(_DWORD *)(v113 + 32) < v114 )
          {
            v113 = *(_QWORD *)(v113 + 24);
          }
          else
          {
            v115 = (int *)v113;
            v113 = *(_QWORD *)(v113 + 16);
          }
        }
        while ( v113 );
        if ( v115 == &v133 || v114 < v115[8] )
        {
LABEL_140:
          if ( v21 != v23 )
          {
            v80 = v23 + 24;
            if ( v21 == v23 + 16 )
              goto LABEL_176;
            if ( v137 )
              goto LABEL_156;
            while ( 1 )
            {
              v81 = v130;
              v82 = &v130[(unsigned int)v131];
              if ( v130 != v82 )
              {
                v83 = *((_DWORD *)v80 - 2);
                while ( *v81 != v83 )
                {
                  if ( v82 == ++v81 )
                    goto LABEL_154;
                }
                if ( v82 != v81 )
                {
LABEL_148:
                  *(_DWORD *)v23 = v83;
                  if ( v80 != v23 + 8 )
                  {
                    v84 = *((_QWORD *)v23 + 1);
                    if ( v84 )
                      sub_161E7C0((__int64)(v23 + 8), v84);
                    v85 = *(unsigned __int8 **)v80;
                    *((_QWORD *)v23 + 1) = *(_QWORD *)v80;
                    if ( v85 )
                    {
                      sub_1623210((__int64)v80, v85, (__int64)(v23 + 8));
                      *(_QWORD *)v80 = 0;
                    }
                  }
                  v23 += 16;
                }
              }
LABEL_154:
              v86 = v80 + 16;
              if ( v21 == v80 + 8 )
                break;
              while ( 1 )
              {
                v80 = v86;
                if ( !v137 )
                  break;
LABEL_156:
                v87 = (__int64)v134;
                if ( !v134 )
                  goto LABEL_154;
                v83 = *((_DWORD *)v80 - 2);
                v88 = &v133;
                do
                {
                  while ( 1 )
                  {
                    v89 = *(_QWORD *)(v87 + 16);
                    v90 = *(_QWORD *)(v87 + 24);
                    if ( *(_DWORD *)(v87 + 32) >= v83 )
                      break;
                    v87 = *(_QWORD *)(v87 + 24);
                    if ( !v90 )
                      goto LABEL_161;
                  }
                  v88 = (int *)v87;
                  v87 = *(_QWORD *)(v87 + 16);
                }
                while ( v89 );
LABEL_161:
                if ( v88 == &v133 )
                  goto LABEL_154;
                if ( v83 >= v88[8] )
                  goto LABEL_148;
                v86 = v80 + 16;
                if ( v21 == v80 + 8 )
                  goto LABEL_164;
              }
            }
LABEL_164:
            v124 = (char *)v123[1];
            v91 = &v124[16 * *((unsigned int *)v123 + 4)] - v21;
            v92 = v91 >> 4;
            if ( v91 <= 0 )
            {
              v21 = &v124[16 * *((unsigned int *)v123 + 4)];
            }
            else
            {
              v93 = (__int64 *)v86;
              v94 = (__int64 *)(v23 + 8);
              do
              {
                *((_DWORD *)v94 - 2) = *((_DWORD *)v93 - 2);
                if ( v94 != v93 )
                {
                  if ( *v94 )
                    sub_161E7C0((__int64)v94, *v94);
                  v95 = (unsigned __int8 *)*v93;
                  *v94 = *v93;
                  if ( v95 )
                  {
                    sub_1623210((__int64)v93, v95, (__int64)v94);
                    *v93 = 0;
                  }
                }
                v93 += 2;
                v94 += 2;
                --v92;
              }
              while ( v92 );
              v23 += v91;
              v124 = (char *)v123[1];
              v21 = &v124[16 * *((unsigned int *)v123 + 4)];
            }
            if ( v21 != v23 )
            {
LABEL_176:
              do
              {
                v96 = *((_QWORD *)v21 - 1);
                v21 -= 16;
                if ( v96 )
                  sub_161E7C0((__int64)(v21 + 8), v96);
              }
              while ( v23 != v21 );
              v124 = (char *)v123[1];
            }
          }
          goto LABEL_178;
        }
LABEL_61:
        v23 = v21;
        goto LABEL_178;
      }
      v110 = (__int64 *)(v16 + 56LL * v17);
      v111 = 1;
      v112 = 0;
      while ( v18 != -8 )
      {
        if ( v18 == -16 && !v112 )
          v112 = v110;
        v17 = (v14 - 1) & (v17 + v111);
        v110 = (__int64 *)(v16 + 56LL * v17);
        v18 = *v110;
        if ( a1 == *v110 )
        {
          v123 = (__int64 *)(v16 + 56LL * v17);
          v19 = v123;
          goto LABEL_25;
        }
        ++v111;
      }
      if ( !v112 )
        v112 = v110;
      ++*(_QWORD *)(v122 + 2704);
      v123 = v112;
      v107 = *(_DWORD *)(v122 + 2720) + 1;
      if ( 4 * v107 < 3 * v14 )
      {
        v106 = a1;
        if ( v14 - *(_DWORD *)(v122 + 2724) - v107 > v14 >> 3 )
          goto LABEL_195;
LABEL_194:
        sub_1624590(v15, v14);
        sub_1621460(v15, &v128, &v129);
        v106 = v128;
        v123 = v129;
        v107 = *(_DWORD *)(v122 + 2720) + 1;
LABEL_195:
        *(_DWORD *)(v122 + 2720) = v107;
        if ( *v123 != -8 )
          --*(_DWORD *)(v122 + 2724);
        v23 = (char *)(v123 + 3);
        *v123 = v106;
        v124 = (char *)(v123 + 3);
        v123[1] = (__int64)(v123 + 3);
        v123[2] = 0x200000000LL;
LABEL_178:
        v97 = (v23 - v124) >> 4;
        *((_DWORD *)v123 + 4) = v97;
        if ( (_DWORD)v97 )
        {
LABEL_111:
          sub_161DBC0((__int64)v134);
          if ( v130 != (unsigned int *)v132 )
            _libc_free((unsigned __int64)v130);
          return;
        }
        v98 = *(_DWORD *)(v122 + 2728);
        if ( v98 )
        {
          v99 = v98 - 1;
          v100 = *(_QWORD *)(v122 + 2712);
          v101 = (v98 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v102 = 1;
          v55 = (__int64 *)(v100 + 56LL * v101);
          v103 = *v55;
          if ( a1 != *v55 )
          {
            while ( v103 != -8 )
            {
              v101 = v99 & (v102 + v101);
              v55 = (__int64 *)(v100 + 56LL * v101);
              v103 = *v55;
              if ( a1 == *v55 )
                goto LABEL_181;
              ++v102;
            }
            goto LABEL_110;
          }
LABEL_181:
          v104 = v55[1];
          v58 = v104 + 16LL * *((unsigned int *)v55 + 4);
          if ( v104 == v58 )
            goto LABEL_107;
          do
          {
            v105 = *(_QWORD *)(v58 - 8);
            v58 -= 16LL;
            if ( v105 )
              sub_161E7C0(v58 + 8, v105);
          }
          while ( v104 != v58 );
LABEL_106:
          v58 = v55[1];
LABEL_107:
          if ( (__int64 *)v58 != v55 + 3 )
            _libc_free(v58);
          *v55 = -16;
          --*(_DWORD *)(v122 + 2720);
          ++*(_DWORD *)(v122 + 2724);
        }
LABEL_110:
        *(_WORD *)(a1 + 18) &= ~0x8000u;
        goto LABEL_111;
      }
    }
    else
    {
      ++*(_QWORD *)(v122 + 2704);
    }
    v14 *= 2;
    goto LABEL_194;
  }
}
