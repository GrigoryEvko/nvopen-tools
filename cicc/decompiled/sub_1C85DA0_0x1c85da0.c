// Function: sub_1C85DA0
// Address: 0x1c85da0
//
char __fastcall sub_1C85DA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r9
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 *v6; // r8
  char v7; // r15
  __int64 v9; // rsi
  __int64 *v10; // rdi
  __int64 *v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // r13
  __int64 *v14; // rbx
  __int64 v15; // rax
  __int64 *v16; // rcx
  __int64 v17; // rdx
  __int64 *v18; // rax
  __int64 *v19; // r12
  __int64 v20; // rsi
  __int64 *v21; // rbx
  __int64 v22; // rdi
  __int64 *v23; // r12
  __int64 *v24; // rbx
  __int64 v25; // rdi
  __int64 *v26; // rax
  __int64 *v27; // rbx
  __int64 v28; // r13
  __int64 *v29; // r12
  __int64 v30; // rax
  __int64 v31; // r12
  int v32; // r13d
  int v33; // ebx
  __int64 v34; // rsi
  char result; // al
  __int64 *v36; // rax
  __int64 *v37; // rax
  __int64 *v38; // rdx
  __int64 *v39; // r15
  __int64 *v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // r8
  __int64 *v43; // rbx
  __int64 v44; // r13
  __int64 *v45; // rax
  _QWORD *v46; // rdi
  _QWORD *v47; // rax
  __int64 *v48; // r14
  __int64 *v49; // rax
  __int64 v50; // r15
  __int64 v51; // rcx
  _QWORD *v52; // rax
  __int64 *v53; // r9
  __int64 *v54; // rax
  __int64 *v55; // rdi
  __int64 v56; // r14
  _QWORD *v57; // rax
  __int64 *v58; // rsi
  __int64 *v59; // rcx
  __int64 *v60; // rsi
  __int64 *v61; // rdx
  __int64 v62; // rdx
  _QWORD *v63; // rax
  __int64 *v64; // rdx
  __int64 v65; // r15
  __int64 v66; // rsi
  __int64 *v67; // r8
  __int64 v68; // r9
  __int64 v69; // r15
  __int64 *v70; // r14
  int v71; // r12d
  __int64 v72; // rbx
  __int64 v73; // rsi
  __int64 *v74; // rax
  __int64 *v75; // r12
  __int64 v76; // r14
  int v77; // r15d
  int v78; // r12d
  __int64 v79; // r13
  __int64 v80; // r14
  __int64 v81; // rsi
  __int64 *v82; // rax
  __int64 *v83; // rax
  __int64 *v84; // r14
  __int64 v85; // r15
  __int64 *v86; // r12
  __int64 *v87; // rax
  char v88; // [rsp+17h] [rbp-1A9h]
  __int64 *v89; // [rsp+18h] [rbp-1A8h]
  __int64 *v90; // [rsp+18h] [rbp-1A8h]
  __int64 *v91; // [rsp+20h] [rbp-1A0h]
  __int64 *v92; // [rsp+20h] [rbp-1A0h]
  __int64 v94; // [rsp+28h] [rbp-198h]
  __int64 *v95; // [rsp+28h] [rbp-198h]
  char v98; // [rsp+38h] [rbp-188h]
  char v99; // [rsp+38h] [rbp-188h]
  __int64 v100; // [rsp+48h] [rbp-178h] BYREF
  __int64 v101; // [rsp+50h] [rbp-170h] BYREF
  __int64 *v102; // [rsp+58h] [rbp-168h]
  __int64 *v103; // [rsp+60h] [rbp-160h]
  __int64 v104; // [rsp+68h] [rbp-158h]
  int v105; // [rsp+70h] [rbp-150h]
  _BYTE v106[24]; // [rsp+78h] [rbp-148h] BYREF
  __int64 v107; // [rsp+90h] [rbp-130h] BYREF
  __int64 *v108; // [rsp+98h] [rbp-128h]
  __int64 *v109; // [rsp+A0h] [rbp-120h]
  __int64 v110; // [rsp+A8h] [rbp-118h]
  int v111; // [rsp+B0h] [rbp-110h]
  _QWORD v112[3]; // [rsp+B8h] [rbp-108h] BYREF
  __int64 v113; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v114; // [rsp+D8h] [rbp-E8h]
  _QWORD *v115; // [rsp+E0h] [rbp-E0h]
  __int64 v116; // [rsp+E8h] [rbp-D8h]
  __int64 v117; // [rsp+F0h] [rbp-D0h]
  unsigned __int64 v118; // [rsp+F8h] [rbp-C8h]
  _QWORD *v119; // [rsp+100h] [rbp-C0h]
  _QWORD *v120; // [rsp+108h] [rbp-B8h]
  _QWORD *v121; // [rsp+110h] [rbp-B0h]
  __int64 *v122; // [rsp+118h] [rbp-A8h]
  __int64 v123; // [rsp+120h] [rbp-A0h] BYREF
  __int64 *v124; // [rsp+128h] [rbp-98h]
  __int64 *v125; // [rsp+130h] [rbp-90h]
  __int64 v126; // [rsp+138h] [rbp-88h]
  int v127; // [rsp+140h] [rbp-80h]
  _BYTE v128[120]; // [rsp+148h] [rbp-78h] BYREF

  v3 = (__int64 *)v106;
  v4 = *(_QWORD *)a3;
  v5 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  v101 = 0;
  v102 = (__int64 *)v106;
  v103 = (__int64 *)v106;
  v104 = 2;
  v105 = 0;
  if ( v5 == v4 )
  {
    v88 = 1;
    goto LABEL_17;
  }
  v6 = (__int64 *)v106;
  v7 = 1;
  do
  {
LABEL_5:
    v9 = *(_QWORD *)(*(_QWORD *)v4 + 40LL);
    if ( *(_QWORD *)(a2 + 40) != v9 )
      v7 = 0;
    if ( v6 != v3 )
    {
LABEL_3:
      sub_16CCBA0((__int64)&v101, v9);
      v6 = v103;
      v3 = v102;
      goto LABEL_4;
    }
    v10 = &v6[HIDWORD(v104)];
    if ( v10 == v6 )
    {
LABEL_163:
      if ( HIDWORD(v104) >= (unsigned int)v104 )
        goto LABEL_3;
      ++HIDWORD(v104);
      *v10 = v9;
      v3 = v102;
      ++v101;
      v6 = v103;
    }
    else
    {
      v11 = v6;
      v12 = 0;
      while ( v9 != *v11 )
      {
        if ( *v11 == -2 )
          v12 = v11;
        if ( v10 == ++v11 )
        {
          if ( !v12 )
            goto LABEL_163;
          v4 += 8;
          *v12 = v9;
          v6 = v103;
          --v105;
          v3 = v102;
          ++v101;
          if ( v4 != v5 )
            goto LABEL_5;
          goto LABEL_16;
        }
      }
    }
LABEL_4:
    v4 += 8;
  }
  while ( v4 != v5 );
LABEL_16:
  v88 = v7;
LABEL_17:
  v123 = 0;
  v124 = (__int64 *)v128;
  v125 = (__int64 *)v128;
  v126 = 8;
  v13 = *(_QWORD *)(a2 + 40);
  v127 = 0;
  v113 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v114 = 8;
  v113 = sub_22077B0(64);
  v14 = (__int64 *)(((4 * v114 - 4) & 0xFFFFFFFFFFFFFFF8LL) + v113);
  v15 = sub_22077B0(512);
  v16 = v112;
  v118 = (unsigned __int64)v14;
  v17 = v15 + 512;
  *v14 = v15;
  v116 = v15;
  v120 = (_QWORD *)v15;
  v115 = (_QWORD *)v15;
  v119 = (_QWORD *)v15;
  v110 = 0x100000002LL;
  v18 = v103;
  v117 = v17;
  v122 = v14;
  v121 = (_QWORD *)v17;
  v108 = v112;
  v109 = v112;
  v111 = 0;
  v112[0] = v13;
  v107 = 1;
  if ( v103 == v102 )
    v19 = &v103[HIDWORD(v104)];
  else
    v19 = &v103[(unsigned int)v104];
  if ( v103 != v19 )
  {
    while ( 1 )
    {
      v20 = *v18;
      v21 = v18;
      if ( (unsigned __int64)*v18 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v19 == ++v18 )
        goto LABEL_22;
    }
    if ( v19 != v18 )
    {
      v39 = v112;
LABEL_116:
      v53 = &v39[HIDWORD(v110)];
      if ( v53 == v39 )
        goto LABEL_161;
      v54 = v39;
      v55 = 0;
      while ( *v54 != v20 )
      {
        if ( *v54 == -2 )
          v55 = v54;
        if ( v53 == ++v54 )
        {
          if ( v55 )
          {
            *v55 = v20;
            v39 = v109;
            --v111;
            v16 = v108;
            ++v107;
            v20 = *v21;
            if ( v13 == *v21 )
              goto LABEL_71;
LABEL_124:
            v56 = *(_QWORD *)(v20 + 8);
            if ( v56 )
            {
              v89 = v16;
              do
              {
                v57 = sub_1648700(v56);
                if ( (unsigned __int8)(*((_BYTE *)v57 + 16) - 25) <= 9u )
                {
                  while ( 1 )
                  {
                    v62 = v57[5];
                    if ( v13 != v62 )
                    {
                      v63 = v119;
                      v100 = v62;
                      if ( v119 == v121 - 1 )
                      {
                        sub_1C85BA0(&v113, &v100);
                      }
                      else
                      {
                        if ( v119 )
                        {
                          *v119 = v62;
                          v63 = v119;
                        }
                        v119 = v63 + 1;
                      }
                    }
                    do
                    {
                      v56 = *(_QWORD *)(v56 + 8);
                      if ( !v56 )
                      {
                        v39 = v109;
                        v16 = v108;
                        goto LABEL_71;
                      }
                      v57 = sub_1648700(v56);
                    }
                    while ( (unsigned __int8)(*((_BYTE *)v57 + 16) - 25) > 9u );
                  }
                }
                v56 = *(_QWORD *)(v56 + 8);
              }
              while ( v56 );
              v16 = v89;
            }
            goto LABEL_71;
          }
LABEL_161:
          if ( HIDWORD(v110) < (unsigned int)v110 )
          {
            ++HIDWORD(v110);
            *v53 = v20;
            v16 = v108;
            ++v107;
            v39 = v109;
            v20 = *v21;
            goto LABEL_70;
          }
          goto LABEL_69;
        }
      }
      while ( 1 )
      {
LABEL_70:
        if ( v13 != v20 )
          goto LABEL_124;
LABEL_71:
        v40 = v21 + 1;
        if ( v21 + 1 == v19 )
          break;
        while ( 1 )
        {
          v20 = *v40;
          v21 = v40;
          if ( (unsigned __int64)*v40 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v19 == ++v40 )
            goto LABEL_74;
        }
        if ( v19 == v40 )
          break;
        if ( v39 == v16 )
          goto LABEL_116;
LABEL_69:
        sub_16CCBA0((__int64)&v107, v20);
        v20 = *v21;
        v39 = v109;
        v16 = v108;
      }
LABEL_74:
      v41 = (__int64)v119;
      v42 = (unsigned __int64)v39;
      if ( v119 != v115 )
      {
        v43 = v39;
        while ( 1 )
        {
          if ( v120 == (_QWORD *)v41 )
            v41 = *(v122 - 1) + 512;
          v44 = *(_QWORD *)(v41 - 8);
          if ( v16 != v43 )
            break;
          v60 = &v16[HIDWORD(v110)];
          if ( v16 == v60 )
            goto LABEL_157;
          v61 = 0;
          do
          {
            if ( v44 == *v16 )
              goto LABEL_80;
            if ( *v16 == -2 )
              v61 = v16;
            ++v16;
          }
          while ( v60 != v16 );
          if ( !v61 )
          {
LABEL_157:
            if ( HIDWORD(v110) >= (unsigned int)v110 )
              break;
            ++HIDWORD(v110);
            *v60 = v44;
            ++v107;
          }
          else
          {
            *v61 = v44;
            --v111;
            ++v107;
          }
LABEL_80:
          v45 = v124;
          if ( v125 != v124 )
            goto LABEL_81;
          v58 = &v124[HIDWORD(v126)];
          if ( v124 == v58 )
            goto LABEL_159;
          v59 = 0;
          do
          {
            if ( v44 == *v45 )
              goto LABEL_82;
            if ( *v45 == -2 )
              v59 = v45;
            ++v45;
          }
          while ( v58 != v45 );
          if ( !v59 )
          {
LABEL_159:
            if ( HIDWORD(v126) >= (unsigned int)v126 )
            {
LABEL_81:
              sub_16CCBA0((__int64)&v123, v44);
            }
            else
            {
              ++HIDWORD(v126);
              *v58 = v44;
              ++v123;
            }
LABEL_82:
            v46 = v119;
            if ( v119 != v120 )
              goto LABEL_83;
            goto LABEL_138;
          }
          *v59 = v44;
          v46 = v119;
          --v127;
          ++v123;
          if ( v119 != v120 )
          {
LABEL_83:
            v119 = v46 - 1;
            goto LABEL_84;
          }
LABEL_138:
          j_j___libc_free_0(v46, 512);
          v120 = (_QWORD *)*--v122;
          v121 = v120 + 64;
          v119 = v120 + 63;
          v44 = *(_QWORD *)(v44 + 8);
          if ( v44 )
          {
            while ( 1 )
            {
              v47 = sub_1648700(v44);
              if ( (unsigned __int8)(*((_BYTE *)v47 + 16) - 25) <= 9u )
                break;
LABEL_84:
              v44 = *(_QWORD *)(v44 + 8);
              if ( !v44 )
                goto LABEL_139;
            }
            v43 = v109;
            while ( 2 )
            {
              v50 = v47[5];
              v49 = v108;
              if ( v43 == v108 )
              {
                v48 = &v43[HIDWORD(v110)];
                if ( v43 == v48 )
                {
                  v64 = v43;
                }
                else
                {
                  do
                  {
                    if ( v50 == *v49 )
                      break;
                    ++v49;
                  }
                  while ( v48 != v49 );
                  v64 = &v43[HIDWORD(v110)];
                }
LABEL_101:
                while ( v64 != v49 )
                {
                  if ( (unsigned __int64)*v49 < 0xFFFFFFFFFFFFFFFELL )
                    goto LABEL_90;
                  ++v49;
                }
                if ( v48 == v49 )
                  goto LABEL_103;
              }
              else
              {
                v48 = &v43[(unsigned int)v110];
                v49 = sub_16CC9F0((__int64)&v107, v50);
                if ( v50 == *v49 )
                {
                  v43 = v109;
                  if ( v109 == v108 )
                    v64 = &v109[HIDWORD(v110)];
                  else
                    v64 = &v109[(unsigned int)v110];
                  goto LABEL_101;
                }
                v43 = v109;
                if ( v109 == v108 )
                {
                  v49 = &v109[HIDWORD(v110)];
                  v64 = v49;
                  goto LABEL_101;
                }
                v49 = &v109[(unsigned int)v110];
LABEL_90:
                if ( v48 == v49 )
                {
LABEL_103:
                  v51 = sub_1648700(v44)[5];
                  v52 = v119;
                  v100 = v51;
                  if ( v119 != v121 - 1 )
                  {
                    if ( v119 )
                    {
                      *v119 = v51;
                      v52 = v119;
                      v43 = v109;
                    }
                    v119 = v52 + 1;
                    v44 = *(_QWORD *)(v44 + 8);
                    if ( !v44 )
                      goto LABEL_107;
                    goto LABEL_92;
                  }
                  sub_1C85BA0(&v113, &v100);
                  v43 = v109;
                }
              }
              do
              {
                v44 = *(_QWORD *)(v44 + 8);
                if ( !v44 )
                  goto LABEL_107;
LABEL_92:
                v47 = sub_1648700(v44);
              }
              while ( (unsigned __int8)(*((_BYTE *)v47 + 16) - 25) > 9u );
              continue;
            }
          }
LABEL_139:
          v43 = v109;
LABEL_107:
          v41 = (__int64)v119;
          v16 = v108;
          if ( v119 == v115 )
          {
            v42 = (unsigned __int64)v43;
            goto LABEL_109;
          }
        }
        sub_16CCBA0((__int64)&v107, v44);
        goto LABEL_80;
      }
LABEL_109:
      if ( v16 != (__int64 *)v42 )
        _libc_free(v42);
    }
  }
LABEL_22:
  v22 = v113;
  if ( v113 )
  {
    v23 = (__int64 *)v118;
    v24 = v122 + 1;
    if ( (unsigned __int64)(v122 + 1) > v118 )
    {
      do
      {
        v25 = *v23++;
        j_j___libc_free_0(v25, 512);
      }
      while ( v24 > v23 );
      v22 = v113;
    }
    j_j___libc_free_0(v22, 8 * v114);
  }
  v26 = v125;
  if ( v125 == v124 )
    v27 = &v125[HIDWORD(v126)];
  else
    v27 = &v125[(unsigned int)v126];
  if ( v125 != v27 )
  {
    while ( 1 )
    {
      v28 = *v26;
      v29 = v26;
      if ( (unsigned __int64)*v26 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v27 == ++v26 )
        goto LABEL_32;
    }
    if ( v27 != v26 )
    {
      v36 = v102;
      if ( v103 == v102 )
        goto LABEL_57;
      while ( 1 )
      {
        v36 = sub_16CC9F0((__int64)&v101, v28);
        if ( *v36 == v28 )
        {
          if ( v103 == v102 )
            v38 = &v103[HIDWORD(v104)];
          else
            v38 = &v103[(unsigned int)v104];
          goto LABEL_61;
        }
        if ( v103 == v102 )
          break;
        while ( 1 )
        {
          v37 = v29 + 1;
          if ( v29 + 1 == v27 )
            goto LABEL_32;
          v28 = *v37;
          for ( ++v29; (unsigned __int64)*v37 >= 0xFFFFFFFFFFFFFFFELL; v29 = v37 )
          {
            if ( v27 == ++v37 )
              goto LABEL_32;
            v28 = *v37;
          }
          if ( v27 == v29 )
            goto LABEL_32;
          v36 = v102;
          if ( v103 != v102 )
            break;
LABEL_57:
          v38 = &v36[HIDWORD(v104)];
          if ( v36 == v38 )
          {
LABEL_65:
            v36 = v38;
          }
          else
          {
            while ( *v36 != v28 )
            {
              if ( v38 == ++v36 )
                goto LABEL_65;
            }
          }
LABEL_61:
          if ( v36 != v38 )
          {
            *v36 = -2;
            ++v105;
          }
        }
      }
      v36 = &v103[HIDWORD(v104)];
      v38 = v36;
      goto LABEL_61;
    }
  }
LABEL_32:
  v30 = **(_QWORD **)(a2 - 24);
  if ( *(_BYTE *)(v30 + 8) == 16 )
    v30 = **(_QWORD **)(v30 + 16);
  v31 = a2 + 24;
  v32 = *(_DWORD *)(v30 + 8) >> 8;
  v33 = *(_DWORD *)(a3 + 8);
  if ( v88 )
  {
    if ( v33 > 0 )
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v31 + 8);
        LOBYTE(v113) = 0;
        v34 = v31 - 24;
        if ( !v31 )
          v34 = 0;
        if ( sub_1C7E8C0(a1, v34, v32, a3, &v113) )
          break;
        if ( (_BYTE)v113 )
        {
          if ( !--v33 )
            goto LABEL_41;
        }
      }
LABEL_182:
      result = !((unsigned __int8)v113 & (v33 == 1));
      goto LABEL_42;
    }
LABEL_41:
    result = 0;
    goto LABEL_42;
  }
  v94 = *(_QWORD *)(a2 + 40) + 40LL;
  if ( v94 == v31 )
  {
LABEL_183:
    v74 = v125;
    if ( v125 == v124 )
      v75 = &v125[HIDWORD(v126)];
    else
      v75 = &v125[(unsigned int)v126];
    if ( v125 != v75 )
    {
      v76 = *v125;
      if ( (unsigned __int64)*v125 < 0xFFFFFFFFFFFFFFFELL )
      {
LABEL_189:
        v92 = v74;
        if ( v75 != v74 )
        {
          v90 = v75;
          v77 = v33;
          v78 = v32;
          while ( 1 )
          {
            v79 = *(_QWORD *)(v76 + 48);
            v80 = v76 + 40;
            if ( v80 != v79 )
              break;
LABEL_201:
            v82 = v92 + 1;
            if ( v92 + 1 != v90 )
            {
              while ( 1 )
              {
                v76 = *v82;
                if ( (unsigned __int64)*v82 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v90 == ++v82 )
                  goto LABEL_204;
              }
              v92 = v82;
              if ( v90 != v82 )
                continue;
            }
LABEL_204:
            v33 = v77;
            v32 = v78;
            goto LABEL_205;
          }
          while ( 1 )
          {
            v81 = v79 - 24;
            if ( !v79 )
              v81 = 0;
            LOBYTE(v113) = 0;
            result = sub_1C7E8C0(a1, v81, v78, a3, &v113);
            if ( result )
              goto LABEL_42;
            v79 = *(_QWORD *)(v79 + 8);
            v77 = ((_BYTE)v113 == 0) + v77 - 1;
            if ( v80 == v79 )
              goto LABEL_201;
          }
        }
      }
      else
      {
        while ( v75 != ++v74 )
        {
          v76 = *v74;
          if ( (unsigned __int64)*v74 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_189;
        }
      }
    }
LABEL_205:
    v83 = v103;
    if ( v103 == v102 )
      v84 = &v103[HIDWORD(v104)];
    else
      v84 = &v103[(unsigned int)v104];
    if ( v103 != v84 )
    {
      while ( 1 )
      {
        v85 = *v83;
        v86 = v83;
        if ( (unsigned __int64)*v83 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v84 == ++v83 )
        {
          result = 0;
          goto LABEL_42;
        }
      }
      if ( v84 != v83 )
      {
        v67 = &v113;
        while ( 1 )
        {
          v68 = *(_QWORD *)(v85 + 48);
          v69 = v85 + 40;
          if ( v69 != v68 )
          {
            v95 = v84;
            v70 = v86;
            v71 = v33;
            v72 = v68;
            do
            {
              v73 = v72 - 24;
              if ( !v72 )
                v73 = 0;
              v91 = v67;
              LOBYTE(v113) = 0;
              result = sub_1C7E8C0(a1, v73, v32, a3, v67);
              v67 = v91;
              if ( result )
              {
                v33 = v71;
                goto LABEL_182;
              }
              v71 = ((_BYTE)v113 == 0) + v71 - 1;
              if ( !v71 )
                goto LABEL_42;
              v72 = *(_QWORD *)(v72 + 8);
            }
            while ( v69 != v72 );
            v33 = v71;
            v86 = v70;
            v84 = v95;
          }
          v87 = v86 + 1;
          if ( v86 + 1 == v84 )
            break;
          while ( 1 )
          {
            v85 = *v87;
            v86 = v87;
            if ( (unsigned __int64)*v87 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v84 == ++v87 )
            {
              result = 0;
              goto LABEL_42;
            }
          }
          if ( v84 == v87 )
          {
            result = 0;
            goto LABEL_42;
          }
        }
      }
    }
    goto LABEL_41;
  }
  v65 = v31;
  while ( 1 )
  {
    v66 = v65 - 24;
    if ( !v65 )
      v66 = 0;
    LOBYTE(v113) = 0;
    result = sub_1C7E8C0(a1, v66, v32, a3, &v113);
    if ( result )
      break;
    v65 = *(_QWORD *)(v65 + 8);
    v33 = ((_BYTE)v113 == 0) + v33 - 1;
    if ( v94 == v65 )
      goto LABEL_183;
  }
LABEL_42:
  if ( v125 != v124 )
  {
    v98 = result;
    _libc_free((unsigned __int64)v125);
    result = v98;
  }
  if ( v103 != v102 )
  {
    v99 = result;
    _libc_free((unsigned __int64)v103);
    return v99;
  }
  return result;
}
