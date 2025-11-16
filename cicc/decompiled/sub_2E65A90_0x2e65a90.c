// Function: sub_2E65A90
// Address: 0x2e65a90
//
void __fastcall sub_2E65A90(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ebx
  _QWORD *v7; // r14
  _QWORD *v8; // r13
  char v9; // di
  __int64 v10; // rsi
  char *v11; // rax
  __int64 v12; // r14
  _QWORD *v13; // r15
  __int64 v14; // r10
  unsigned int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // edx
  __int64 v19; // r13
  __int64 *v20; // r12
  __int64 *v21; // r8
  __int64 v22; // rbx
  char *v23; // rax
  char *v24; // rdx
  __int64 v25; // r14
  __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // rbx
  unsigned int v29; // eax
  __int64 v30; // rax
  _QWORD *v31; // rbx
  __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rcx
  __int64 v36; // r13
  _QWORD *v37; // rax
  __int64 v38; // rdx
  unsigned __int64 v39; // r8
  __int64 v40; // r9
  _QWORD *v41; // r12
  int v42; // eax
  __int64 v43; // rsi
  __int64 v44; // r10
  unsigned __int64 v45; // rax
  __int64 v46; // rsi
  unsigned __int64 v47; // r14
  unsigned __int64 v48; // rdi
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rcx
  unsigned int v54; // eax
  __int64 v55; // r13
  __int64 v56; // r15
  _QWORD *v57; // rdi
  __int64 v58; // rsi
  _QWORD *v59; // rax
  __int64 v60; // r8
  __int64 v61; // r9
  int v62; // r10d
  __int64 v63; // rax
  __int64 v64; // rdx
  _QWORD *v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 *v68; // r13
  __int64 *v69; // r14
  __int64 v70; // r15
  _QWORD *v71; // rax
  unsigned __int64 v72; // r12
  __int64 *v73; // rax
  __int64 v74; // rcx
  __int64 v75; // rdi
  __int64 v76; // r14
  _QWORD *v77; // rax
  _QWORD *v78; // r14
  __int64 v79; // rax
  __int64 v80; // rdx
  int v81; // r11d
  __int64 v82; // r9
  _QWORD *v83; // rdi
  __int64 v84; // r8
  unsigned int v85; // eax
  __int64 v86; // rbx
  __int64 v87; // rdx
  __int64 v88; // rax
  _QWORD *v89; // rdi
  int v90; // r12d
  _QWORD *v91; // rcx
  __int64 *v92; // rdx
  __int64 v93; // r14
  __int64 v94; // rbx
  unsigned __int64 v95; // rdi
  __int64 v96; // [rsp+0h] [rbp-3D0h]
  __int64 v97; // [rsp+8h] [rbp-3C8h]
  int v98; // [rsp+14h] [rbp-3BCh]
  __int64 v99; // [rsp+18h] [rbp-3B8h]
  unsigned __int64 v101; // [rsp+28h] [rbp-3A8h]
  int v102; // [rsp+30h] [rbp-3A0h]
  unsigned int v103; // [rsp+38h] [rbp-398h]
  int v104; // [rsp+38h] [rbp-398h]
  __int64 v105; // [rsp+38h] [rbp-398h]
  _QWORD *v107; // [rsp+48h] [rbp-388h]
  __int64 v108; // [rsp+50h] [rbp-380h]
  __int64 v109; // [rsp+50h] [rbp-380h]
  __int64 v110; // [rsp+50h] [rbp-380h]
  _QWORD *v111; // [rsp+50h] [rbp-380h]
  unsigned int v112; // [rsp+58h] [rbp-378h]
  __int64 *v113; // [rsp+58h] [rbp-378h]
  __int64 *v114; // [rsp+58h] [rbp-378h]
  unsigned __int64 v115; // [rsp+68h] [rbp-368h] BYREF
  __int64 v116; // [rsp+70h] [rbp-360h] BYREF
  char *v117; // [rsp+78h] [rbp-358h]
  __int64 v118; // [rsp+80h] [rbp-350h]
  int v119; // [rsp+88h] [rbp-348h]
  char v120; // [rsp+8Ch] [rbp-344h]
  char v121; // [rsp+90h] [rbp-340h] BYREF
  _QWORD *v122; // [rsp+190h] [rbp-240h] BYREF
  __int64 v123; // [rsp+198h] [rbp-238h]
  _QWORD v124[70]; // [rsp+1A0h] [rbp-230h] BYREF

  if ( *(_QWORD *)(a1 + 544) )
  {
    v6 = a3;
    if ( a3 )
    {
      v116 = 0;
      v117 = &v121;
      v7 = &a2[3 * a3];
      v120 = 1;
      v118 = 32;
      v119 = 0;
      v107 = v7;
      if ( a2 != v7 )
      {
        v8 = a2;
        v9 = 1;
        do
        {
          while ( 1 )
          {
            while ( 1 )
            {
              v10 = v8[2];
              if ( v9 )
                break;
LABEL_106:
              v8 += 3;
              sub_C8CC70((__int64)&v116, v10, a3, a4, a5, a6);
              v9 = v120;
              if ( v8 == v7 )
                goto LABEL_11;
            }
            v11 = v117;
            a4 = HIDWORD(v118);
            a3 = (__int64)&v117[8 * HIDWORD(v118)];
            if ( v117 != (char *)a3 )
              break;
LABEL_109:
            if ( HIDWORD(v118) >= (unsigned int)v118 )
              goto LABEL_106;
            a4 = (unsigned int)(HIDWORD(v118) + 1);
            v8 += 3;
            ++HIDWORD(v118);
            *(_QWORD *)a3 = v10;
            v9 = v120;
            ++v116;
            if ( v8 == v7 )
              goto LABEL_11;
          }
          while ( v10 != *(_QWORD *)v11 )
          {
            v11 += 8;
            if ( (char *)a3 == v11 )
              goto LABEL_109;
          }
          v8 += 3;
        }
        while ( v8 != v7 );
LABEL_11:
        sub_B48880((__int64 *)&v115, v6, 1u);
        v103 = 0;
        v12 = *(_QWORD *)(a1 + 544);
        v13 = a2;
        v14 = a1;
        v15 = *(_DWORD *)(v12 + 32);
LABEL_12:
        v16 = v13[1];
        if ( v16 )
        {
          v17 = (unsigned int)(*(_DWORD *)(v16 + 24) + 1);
          v18 = *(_DWORD *)(v16 + 24) + 1;
        }
        else
        {
          v17 = 0;
          v18 = 0;
        }
        v19 = 0;
        if ( v15 > v18 )
          v19 = *(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v17);
        v20 = *(__int64 **)(v16 + 64);
        v21 = &v20[*(unsigned int *)(v16 + 72)];
        if ( v20 == v21 )
          goto LABEL_42;
        while ( 1 )
        {
          v22 = *v20;
          if ( v13[2] == *v20 )
            goto LABEL_101;
          if ( v120 )
            break;
          v108 = v14;
          v113 = v21;
          v73 = sub_C8CA60((__int64)&v116, v22);
          v21 = v113;
          v14 = v108;
          if ( v73 )
            goto LABEL_23;
          v25 = *(_QWORD *)(v108 + 544);
          if ( v22 )
          {
LABEL_25:
            v26 = (unsigned int)(*(_DWORD *)(v22 + 24) + 1);
            v27 = *(_DWORD *)(v22 + 24) + 1;
            goto LABEL_26;
          }
LABEL_105:
          v26 = 0;
          v27 = 0;
LABEL_26:
          if ( v27 >= *(_DWORD *)(v25 + 32) )
            goto LABEL_101;
          v28 = *(_QWORD *)(*(_QWORD *)(v25 + 24) + 8 * v26);
          if ( !v28 || v19 == v28 )
            goto LABEL_101;
          if ( !v19 )
            goto LABEL_39;
          if ( v19 == *(_QWORD *)(v28 + 8) )
            goto LABEL_101;
          if ( v28 == *(_QWORD *)(v19 + 8) || *(_DWORD *)(v19 + 16) >= *(_DWORD *)(v28 + 16) )
            goto LABEL_39;
          if ( *(_BYTE *)(v25 + 112) )
            goto LABEL_99;
          v29 = *(_DWORD *)(v25 + 116) + 1;
          *(_DWORD *)(v25 + 116) = v29;
          if ( v29 > 0x20 )
          {
            HIDWORD(v123) = 32;
            v122 = v124;
            v79 = *(_QWORD *)(v25 + 96);
            if ( v79 )
            {
              v80 = *(_QWORD *)(v79 + 24);
              v81 = 1;
              v82 = v28;
              v124[0] = *(_QWORD *)(v25 + 96);
              v114 = v21;
              v83 = v124;
              v84 = (__int64)v20;
              v124[1] = v80;
              LODWORD(v123) = 1;
              v109 = v14;
              *(_DWORD *)(v79 + 72) = 0;
              v85 = 1;
              do
              {
                v90 = v81++;
                v91 = &v83[2 * v85 - 2];
                v92 = (__int64 *)v91[1];
                if ( v92 == (__int64 *)(*(_QWORD *)(*v91 + 24LL) + 8LL * *(unsigned int *)(*v91 + 32LL)) )
                {
                  --v85;
                  *(_DWORD *)(*v91 + 76LL) = v90;
                  LODWORD(v123) = v85;
                }
                else
                {
                  v86 = *v92;
                  v91[1] = v92 + 1;
                  v87 = (unsigned int)v123;
                  v88 = *(_QWORD *)(v86 + 24);
                  if ( (unsigned __int64)(unsigned int)v123 + 1 > HIDWORD(v123) )
                  {
                    v96 = v82;
                    v97 = v84;
                    v98 = v81;
                    v99 = *(_QWORD *)(v86 + 24);
                    sub_C8D5F0((__int64)&v122, v124, (unsigned int)v123 + 1LL, 0x10u, v84, v82);
                    v83 = v122;
                    v87 = (unsigned int)v123;
                    v82 = v96;
                    v84 = v97;
                    v81 = v98;
                    v88 = v99;
                  }
                  v89 = &v83[2 * v87];
                  *v89 = v86;
                  v89[1] = v88;
                  v85 = v123 + 1;
                  LODWORD(v123) = v123 + 1;
                  *(_DWORD *)(v86 + 72) = v90;
                  v83 = v122;
                }
              }
              while ( v85 );
              v20 = (__int64 *)v84;
              *(_BYTE *)(v25 + 112) = 1;
              v28 = v82;
              v21 = v114;
              *(_DWORD *)(v25 + 116) = 0;
              v14 = v109;
              if ( v83 != v124 )
              {
                _libc_free((unsigned __int64)v83);
                v21 = v114;
                v14 = v109;
              }
            }
LABEL_99:
            if ( *(_DWORD *)(v28 + 72) < *(_DWORD *)(v19 + 72) || *(_DWORD *)(v28 + 76) > *(_DWORD *)(v19 + 76) )
              goto LABEL_39;
            goto LABEL_101;
          }
          do
          {
            v30 = v28;
            v28 = *(_QWORD *)(v28 + 8);
          }
          while ( v28 && *(_DWORD *)(v19 + 16) <= *(_DWORD *)(v28 + 16) );
          if ( v19 != v30 )
          {
LABEL_39:
            if ( (v115 & 1) != 0 )
              v115 = 2 * ((v115 >> 58 << 57) | ~(-1LL << (v115 >> 58)) & (v115 >> 1) & ~(1LL << v103)) + 1;
            else
              *(_QWORD *)(*(_QWORD *)v115 + 8LL * (v103 >> 6)) &= ~(1LL << v103);
LABEL_41:
            v12 = *(_QWORD *)(v14 + 544);
            v15 = *(_DWORD *)(v12 + 32);
LABEL_42:
            ++v103;
            v13 += 3;
            if ( v107 == v13 )
            {
              v31 = a2;
              v32 = v12;
              v112 = 0;
              v33 = *a2;
              v34 = a2[2];
              if ( !*a2 )
                goto LABEL_86;
LABEL_44:
              v35 = (unsigned int)(*(_DWORD *)(v33 + 24) + 1);
              if ( *(_DWORD *)(v33 + 24) + 1 >= v15 )
              {
                while ( 1 )
                {
                  *(_BYTE *)(v32 + 112) = 0;
                  v71 = (_QWORD *)sub_22077B0(0x50u);
                  v41 = v71;
                  if ( v71 )
                    break;
                  v36 = 0;
LABEL_49:
                  if ( v34 )
                  {
                    v43 = (unsigned int)(*(_DWORD *)(v34 + 24) + 1);
                    v44 = 8 * v43;
                  }
                  else
                  {
                    v44 = 0;
                    LODWORD(v43) = 0;
                  }
                  v45 = *(unsigned int *)(v32 + 32);
                  if ( (unsigned int)v45 > (unsigned int)v43 )
                    goto LABEL_52;
                  v74 = (unsigned int)(v43 + 1);
                  v75 = (__int64)(*(_QWORD *)(*(_QWORD *)(v32 + 104) + 104LL)
                                - *(_QWORD *)(*(_QWORD *)(v32 + 104) + 96LL)) >> 3;
                  if ( (unsigned int)v74 <= (unsigned int)v75 )
                    v74 = (unsigned int)v75;
                  if ( (unsigned int)v74 == v45 )
                  {
LABEL_52:
                    v46 = *(_QWORD *)(v32 + 24);
                  }
                  else
                  {
                    v76 = 8LL * (unsigned int)v74;
                    if ( (unsigned int)v74 < v45 )
                    {
                      v46 = *(_QWORD *)(v32 + 24);
                      v93 = v46 + v76;
                      if ( v46 + 8 * v45 != v93 )
                      {
                        v105 = v44;
                        v102 = v74;
                        v111 = v31;
                        v94 = v46 + 8 * v45;
                        do
                        {
                          v39 = *(_QWORD *)(v94 - 8);
                          v94 -= 8;
                          if ( v39 )
                          {
                            v95 = *(_QWORD *)(v39 + 24);
                            if ( v95 != v39 + 40 )
                            {
                              v101 = v39;
                              _libc_free(v95);
                              v39 = v101;
                            }
                            j_j___libc_free_0(v39);
                          }
                        }
                        while ( v93 != v94 );
                        v31 = v111;
                        v44 = v105;
                        LODWORD(v74) = v102;
                        v46 = *(_QWORD *)(v32 + 24);
                      }
                    }
                    else
                    {
                      if ( (unsigned int)v74 > (unsigned __int64)*(unsigned int *)(v32 + 36) )
                      {
                        v104 = v74;
                        v110 = v44;
                        sub_239B9C0(v32 + 24, (unsigned int)v74, v38, v74, v39, v40);
                        v45 = *(unsigned int *)(v32 + 32);
                        LODWORD(v74) = v104;
                        v44 = v110;
                      }
                      v46 = *(_QWORD *)(v32 + 24);
                      v77 = (_QWORD *)(v46 + 8 * v45);
                      v78 = (_QWORD *)(v46 + v76);
                      if ( v77 != v78 )
                      {
                        do
                        {
                          if ( v77 )
                            *v77 = 0;
                          ++v77;
                        }
                        while ( v78 != v77 );
                        v46 = *(_QWORD *)(v32 + 24);
                      }
                    }
                    *(_DWORD *)(v32 + 32) = v74;
                  }
                  v47 = *(_QWORD *)(v46 + v44);
                  *(_QWORD *)(v46 + v44) = v41;
                  if ( v47 )
                  {
                    v48 = *(_QWORD *)(v47 + 24);
                    if ( v48 != v47 + 40 )
                      _libc_free(v48);
                    j_j___libc_free_0(v47);
                  }
                  if ( v36 )
                  {
                    v49 = *(unsigned int *)(v36 + 32);
                    if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 36) )
                    {
                      sub_C8D5F0(v36 + 24, (const void *)(v36 + 40), v49 + 1, 8u, v39, v40);
                      v49 = *(unsigned int *)(v36 + 32);
                    }
                    *(_QWORD *)(*(_QWORD *)(v36 + 24) + 8 * v49) = v41;
                    ++*(_DWORD *)(v36 + 32);
                  }
                  if ( (v115 & 1) != 0 )
                    v50 = (((v115 >> 1) & ~(-1LL << (v115 >> 58))) >> v112) & 1;
                  else
                    v50 = (*(_QWORD *)(*(_QWORD *)v115 + 8LL * (v112 >> 6)) >> v112) & 1LL;
                  if ( (_BYTE)v50 )
                  {
                    v51 = *(_QWORD *)(a1 + 544);
                    v52 = v31[1];
                    if ( v52 )
                    {
                      v53 = (unsigned int)(*(_DWORD *)(v52 + 24) + 1);
                      v54 = *(_DWORD *)(v52 + 24) + 1;
                    }
                    else
                    {
                      v53 = 0;
                      v54 = 0;
                    }
                    if ( v54 >= *(_DWORD *)(v51 + 32) )
                    {
                      *(_BYTE *)(v51 + 112) = 0;
                      BUG();
                    }
                    v55 = *(_QWORD *)(*(_QWORD *)(v51 + 24) + 8 * v53);
                    *(_BYTE *)(v51 + 112) = 0;
                    v56 = *(_QWORD *)(v55 + 8);
                    if ( v41 != (_QWORD *)v56 )
                    {
                      v122 = (_QWORD *)v55;
                      v57 = *(_QWORD **)(v56 + 24);
                      v58 = (__int64)&v57[*(unsigned int *)(v56 + 32)];
                      v59 = sub_2E641C0(v57, v58, (__int64 *)&v122);
                      if ( v59 + 1 != (_QWORD *)v58 )
                      {
                        memmove(v59, v59 + 1, v58 - (_QWORD)(v59 + 1));
                        v62 = *(_DWORD *)(v56 + 32);
                      }
                      *(_DWORD *)(v56 + 32) = v62 - 1;
                      *(_QWORD *)(v55 + 8) = v41;
                      v63 = *((unsigned int *)v41 + 8);
                      if ( v63 + 1 > (unsigned __int64)*((unsigned int *)v41 + 9) )
                      {
                        sub_C8D5F0((__int64)(v41 + 3), v41 + 5, v63 + 1, 8u, v60, v61);
                        v63 = *((unsigned int *)v41 + 8);
                      }
                      *(_QWORD *)(v41[3] + 8 * v63) = v55;
                      ++*((_DWORD *)v41 + 8);
                      if ( *(_DWORD *)(v55 + 16) != *(_DWORD *)(*(_QWORD *)(v55 + 8) + 16LL) + 1 )
                      {
                        v124[0] = v55;
                        LODWORD(v64) = 1;
                        v122 = v124;
                        v65 = v124;
                        v123 = 0x4000000001LL;
                        do
                        {
                          v66 = (unsigned int)v64;
                          v64 = (unsigned int)(v64 - 1);
                          v67 = v65[v66 - 1];
                          LODWORD(v123) = v64;
                          v68 = *(__int64 **)(v67 + 24);
                          *(_DWORD *)(v67 + 16) = *(_DWORD *)(*(_QWORD *)(v67 + 8) + 16LL) + 1;
                          v69 = &v68[*(unsigned int *)(v67 + 32)];
                          if ( v68 != v69 )
                          {
                            do
                            {
                              v70 = *v68;
                              if ( *(_DWORD *)(*v68 + 16) != *(_DWORD *)(*(_QWORD *)(*v68 + 8) + 16LL) + 1 )
                              {
                                if ( v64 + 1 > (unsigned __int64)HIDWORD(v123) )
                                {
                                  sub_C8D5F0((__int64)&v122, v124, v64 + 1, 8u, v64 + 1, v61);
                                  v64 = (unsigned int)v123;
                                }
                                v122[v64] = v70;
                                v64 = (unsigned int)(v123 + 1);
                                LODWORD(v123) = v123 + 1;
                              }
                              ++v68;
                            }
                            while ( v69 != v68 );
                            v65 = v122;
                          }
                        }
                        while ( (_DWORD)v64 );
                        if ( v65 != v124 )
                          _libc_free((unsigned __int64)v65);
                      }
                    }
                  }
                  ++v112;
                  v31 += 3;
                  if ( v107 == v31 )
                    goto LABEL_90;
                  v34 = v31[2];
                  v32 = *(_QWORD *)(a1 + 544);
                  v33 = *v31;
                  v15 = *(_DWORD *)(v32 + 32);
                  if ( *v31 )
                    goto LABEL_44;
LABEL_86:
                  v35 = 0;
                  if ( v15 )
                    goto LABEL_45;
                }
                *v71 = v34;
                v36 = 0;
                v42 = 0;
                v41[1] = 0;
              }
              else
              {
LABEL_45:
                v36 = *(_QWORD *)(*(_QWORD *)(v32 + 24) + 8 * v35);
                *(_BYTE *)(v32 + 112) = 0;
                v37 = (_QWORD *)sub_22077B0(0x50u);
                v41 = v37;
                if ( !v37 )
                  goto LABEL_49;
                *v37 = v34;
                v37[1] = v36;
                if ( v36 )
                  v42 = *(_DWORD *)(v36 + 16) + 1;
                else
                  v42 = 0;
              }
              *((_DWORD *)v41 + 4) = v42;
              v41[3] = v41 + 5;
              v41[4] = 0x400000000LL;
              v41[9] = -1;
              goto LABEL_49;
            }
            goto LABEL_12;
          }
LABEL_101:
          if ( v21 == ++v20 )
            goto LABEL_41;
        }
        v23 = v117;
        v24 = &v117[8 * HIDWORD(v118)];
        if ( v117 != v24 )
        {
          while ( v22 != *(_QWORD *)v23 )
          {
            v23 += 8;
            if ( v24 == v23 )
              goto LABEL_24;
          }
LABEL_23:
          v22 = **(_QWORD **)(v22 + 64);
        }
LABEL_24:
        v25 = *(_QWORD *)(v14 + 544);
        if ( v22 )
          goto LABEL_25;
        goto LABEL_105;
      }
      sub_B48880((__int64 *)&v115, a3, 1u);
LABEL_90:
      v72 = v115;
      if ( (v115 & 1) == 0 && v115 )
      {
        if ( *(_QWORD *)v115 != v115 + 16 )
          _libc_free(*(_QWORD *)v115);
        j_j___libc_free_0(v72);
      }
      if ( !v120 )
        _libc_free((unsigned __int64)v117);
    }
  }
}
