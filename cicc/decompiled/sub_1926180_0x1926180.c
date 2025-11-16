// Function: sub_1926180
// Address: 0x1926180
//
void __fastcall sub_1926180(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 *v4; // rax
  __m128i *v5; // r12
  __int64 v6; // rax
  char *v7; // r13
  unsigned __int64 v8; // r15
  char *v9; // r14
  __int64 v10; // rax
  char *v11; // rdx
  signed __int64 v12; // rax
  int v13; // esi
  int v14; // ecx
  char *v15; // r8
  char *v16; // rax
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // r13
  unsigned __int64 v20; // rax
  __int64 v21; // r14
  char v22; // al
  __int64 v23; // r11
  unsigned int v24; // eax
  char **v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r8
  int v30; // r9d
  __int64 v31; // r12
  unsigned int v32; // ebx
  unsigned int v33; // r15d
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 *v37; // rax
  char *v38; // rdi
  __int64 v39; // r12
  __int64 v40; // rax
  unsigned __int64 v41; // r12
  __int64 v42; // rbx
  char **v43; // r8
  __int64 v44; // rdx
  char **v45; // r14
  __int64 v46; // r13
  __int64 v47; // rdx
  __int64 v48; // rax
  int v49; // ecx
  char *v50; // rbx
  char *v51; // rax
  char v52; // al
  int v53; // eax
  int v54; // eax
  __int64 v55; // rcx
  int v56; // r10d
  __int64 v57; // r8
  unsigned int v58; // edx
  __int64 *v59; // rax
  __int64 v60; // rsi
  unsigned int v61; // edi
  unsigned int v62; // edx
  __int64 *v63; // rax
  __int64 v64; // rcx
  int v65; // esi
  char *v66; // rdi
  __int64 *v67; // rdx
  __int64 v68; // rax
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r12
  __int64 v75; // r13
  __int64 v76; // rax
  __int64 v77; // rcx
  char **v78; // rsi
  __int64 v79; // rdi
  __int64 v80; // r12
  __int64 v81; // rbx
  unsigned __int64 v82; // rdi
  char *v83; // rdi
  int v84; // eax
  int v85; // esi
  int v86; // eax
  int v87; // eax
  int v88; // edi
  int v89; // eax
  __int64 *v90; // [rsp+10h] [rbp-170h]
  __int64 *v93; // [rsp+28h] [rbp-158h]
  __int64 *v94; // [rsp+28h] [rbp-158h]
  __int64 v95; // [rsp+30h] [rbp-150h]
  __int64 v96; // [rsp+48h] [rbp-138h]
  __int64 v97; // [rsp+50h] [rbp-130h]
  __int64 v98; // [rsp+58h] [rbp-128h]
  __int64 v99; // [rsp+60h] [rbp-120h]
  __int64 v100; // [rsp+68h] [rbp-118h]
  __int64 v101; // [rsp+68h] [rbp-118h]
  __int64 v102; // [rsp+70h] [rbp-110h]
  int v104; // [rsp+80h] [rbp-100h]
  __int64 v105; // [rsp+80h] [rbp-100h]
  char *v106; // [rsp+88h] [rbp-F8h]
  int v107; // [rsp+88h] [rbp-F8h]
  __int64 v108; // [rsp+88h] [rbp-F8h]
  char *v109; // [rsp+90h] [rbp-F0h]
  int v110; // [rsp+98h] [rbp-E8h]
  __int64 v111; // [rsp+98h] [rbp-E8h]
  int v112; // [rsp+98h] [rbp-E8h]
  __int64 v113; // [rsp+D0h] [rbp-B0h] BYREF
  char *v114; // [rsp+D8h] [rbp-A8h] BYREF
  __int64 v115; // [rsp+E0h] [rbp-A0h]
  _BYTE v116[40]; // [rsp+E8h] [rbp-98h] BYREF
  char **v117; // [rsp+110h] [rbp-70h] BYREF
  __int64 v118; // [rsp+118h] [rbp-68h]
  char *v119[12]; // [rsp+120h] [rbp-60h] BYREF

  if ( *(_DWORD *)(a2 + 16) )
  {
    v4 = *(__int64 **)(a2 + 8);
    v90 = &v4[9 * *(unsigned int *)(a2 + 24)];
    if ( v4 != v90 )
    {
      while ( 1 )
      {
        v93 = v4;
        if ( *v4 != -8 && *v4 != -16 )
          break;
        v4 += 9;
        if ( v90 == v4 )
          return;
      }
      v102 = *v4;
      if ( v90 != v4 )
      {
        while ( 1 )
        {
          v5 = (__m128i *)v93[1];
          v6 = 24LL * *((unsigned int *)v93 + 4);
          v7 = &v5->m128i_i8[v6];
          sub_1923500((__int64 *)&v117, v5, 0xAAAAAAAAAAAAAAABLL * (v6 >> 3));
          if ( v119[0] )
            sub_19213A0(v5->m128i_i8, v7, v119[0], v118);
          else
            sub_1920B30(v5->m128i_i8, (__int64)v7);
          j_j___libc_free_0(v119[0], 24 * v118);
          v8 = sub_157EBA0(v102);
          v9 = (char *)v93[1];
          v10 = 24LL * *((unsigned int *)v93 + 4);
          v11 = &v9[v10];
          v12 = 0xAAAAAAAAAAAAAAABLL * (v10 >> 3);
          if ( v12 >> 2 )
          {
            v13 = *(_DWORD *)v9;
            v14 = *((_DWORD *)v9 + 1);
            v15 = (char *)v93[1];
            v16 = &v9[96 * (v12 >> 2)];
            while ( 1 )
            {
              if ( *((_DWORD *)v15 + 1) != v14 )
              {
LABEL_14:
                v109 = v15;
                goto LABEL_15;
              }
              v83 = v15 + 24;
              if ( v13 != *((_DWORD *)v15 + 6)
                || v14 != *((_DWORD *)v15 + 7)
                || (v83 = v15 + 48, v13 != *((_DWORD *)v15 + 12))
                || v14 != *((_DWORD *)v15 + 13)
                || (v83 = v15 + 72, v13 != *((_DWORD *)v15 + 18))
                || v14 != *((_DWORD *)v15 + 19) )
              {
                v109 = v83;
                goto LABEL_15;
              }
              v15 += 96;
              if ( v16 == v15 )
                break;
              if ( v13 != *(_DWORD *)v15 )
                goto LABEL_14;
            }
            v109 = v15;
            v12 = 0xAAAAAAAAAAAAAAABLL * ((v11 - v15) >> 3);
          }
          else
          {
            v109 = (char *)v93[1];
          }
          if ( v12 == 2 )
            break;
          if ( v12 == 3 )
          {
            v89 = *(_DWORD *)v9;
            if ( *(_DWORD *)v109 != *(_DWORD *)v9 || *((_DWORD *)v109 + 1) != *((_DWORD *)v9 + 1) )
              goto LABEL_15;
            v109 += 24;
            goto LABEL_174;
          }
          if ( v12 != 1 )
            goto LABEL_170;
          v89 = *(_DWORD *)v9;
LABEL_177:
          if ( *(_DWORD *)v109 != v89 )
            goto LABEL_15;
          if ( *((_DWORD *)v109 + 1) != *((_DWORD *)v9 + 1) )
            v11 = v109;
LABEL_170:
          v109 = v11;
LABEL_15:
          if ( v9 == v109 )
            goto LABEL_98;
          v17 = v8;
          v18 = v93[1];
          while ( 2 )
          {
            v95 = v17;
            v117 = v119;
            v118 = 0x200000000LL;
            LODWORD(v113) = dword_4FAF220;
            do
            {
              while ( 1 )
              {
                v26 = *(_QWORD *)(v18 + 16);
                v110 = *(_DWORD *)v18;
                v104 = *(_DWORD *)(v18 + 4);
                v106 = *(char **)(v18 + 8);
                if ( !v26 )
                  goto LABEL_30;
                if ( a3 == 1 )
                  break;
                v19 = sub_1422850(*(_QWORD *)(a1 + 248), v26);
                v20 = sub_157EBA0(v102);
                if ( v26 == v20 )
                  goto LABEL_26;
                v21 = *(_QWORD *)(v20 + 40);
                v98 = v20;
                v99 = *(_QWORD *)(v19 - 24);
                v96 = *(_QWORD *)(v26 + 40);
                v100 = *(_QWORD *)(v99 + 64);
                v97 = *(_QWORD *)(v19 + 64);
                if ( !sub_15CC890(*(_QWORD *)(a1 + 216), v21, v100) )
                {
                  if ( v21 != v100
                    || v99 == *(_QWORD *)(*(_QWORD *)(a1 + 248) + 120LL)
                    || (unsigned int)*(unsigned __int8 *)(v99 + 16) - 21 > 1 )
                  {
                    goto LABEL_21;
                  }
                  v54 = *(_DWORD *)(a1 + 288);
                  if ( v54 )
                  {
                    v55 = *(_QWORD *)(v99 + 72);
                    v56 = v54 - 1;
                    v57 = *(_QWORD *)(a1 + 272);
                    v58 = (v54 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                    v59 = (__int64 *)(v57 + 16LL * v58);
                    v60 = *v59;
                    if ( v55 == *v59 )
                    {
LABEL_84:
                      v61 = *((_DWORD *)v59 + 2);
                    }
                    else
                    {
                      v86 = 1;
                      while ( v60 != -8 )
                      {
                        v88 = v86 + 1;
                        v58 = v56 & (v86 + v58);
                        v59 = (__int64 *)(v57 + 16LL * v58);
                        v60 = *v59;
                        if ( v55 == *v59 )
                          goto LABEL_84;
                        v86 = v88;
                      }
                      v61 = 0;
                    }
                    v62 = v56 & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
                    v63 = (__int64 *)(v57 + 16LL * v62);
                    v64 = *v63;
                    if ( v98 != *v63 )
                    {
                      v84 = 1;
                      while ( v64 != -8 )
                      {
                        v85 = v84 + 1;
                        v62 = v56 & (v84 + v62);
                        v63 = (__int64 *)(v57 + 16LL * v62);
                        v64 = *v63;
                        if ( v98 == *v63 )
                          goto LABEL_86;
                        v84 = v85;
                      }
                      goto LABEL_30;
                    }
LABEL_86:
                    if ( *((_DWORD *)v63 + 2) > v61 )
                    {
LABEL_21:
                      if ( a3 == 3 )
                      {
                        if ( *(_BYTE *)(v19 + 16) != 22 )
                          v19 = 0;
                        v22 = sub_1924DD0(a1, v98, v19, &v113);
                        v23 = v100;
                        if ( v22 )
                          goto LABEL_30;
                        if ( v21 != v97 )
                          goto LABEL_26;
LABEL_72:
                        sub_15CC890(*(_QWORD *)(a1 + 216), v23, v21);
                        v24 = v118;
                        if ( (unsigned int)v118 < HIDWORD(v118) )
                          goto LABEL_27;
                        goto LABEL_73;
                      }
                      v52 = sub_19258D0(a1, v21, v96, &v113);
                      v23 = v100;
                      if ( !v52 )
                      {
                        if ( v21 == v97 )
                          goto LABEL_72;
LABEL_26:
                        v24 = v118;
                        if ( (unsigned int)v118 < HIDWORD(v118) )
                        {
LABEL_27:
                          v25 = &v117[3 * v24];
                          if ( v25 )
                          {
                            *(_DWORD *)v25 = v110;
                            *((_DWORD *)v25 + 1) = v104;
                            v25[2] = (char *)v26;
                            v25[1] = v106;
                            v24 = v118;
                          }
                          LODWORD(v118) = v24 + 1;
                          goto LABEL_30;
                        }
LABEL_73:
                        sub_1923080((__int64)&v117, 0);
                        v24 = v118;
                        goto LABEL_27;
                      }
                    }
                  }
                }
LABEL_30:
                v18 += 24;
                if ( v109 == (char *)v18 )
                  goto LABEL_35;
              }
              if ( !(unsigned __int8)sub_19258D0(a1, v102, *(_QWORD *)(v26 + 40), &v113) )
                goto LABEL_26;
              v18 += 24;
            }
            while ( v109 != (char *)v18 );
LABEL_35:
            v17 = v95;
            v27 = (__int64)v117;
            v111 = (__int64)v117;
            v28 = 24LL * (unsigned int)v118;
            LODWORD(v29) = sub_15F4D60(v95);
            if ( (unsigned int)v29 > -1431655765 * (unsigned int)(v28 >> 3) )
              goto LABEL_62;
            v105 = v27 + v28;
            if ( v27 != v27 + v28 )
            {
              v101 = v18;
              while ( 1 )
              {
                v31 = *(_QWORD *)(v111 + 8);
                v107 = sub_15F4D60(v95);
                if ( v107 >> 2 > 0 )
                {
                  v32 = 0;
                  while ( v31 != sub_15F4DF0(v95, v32) )
                  {
                    v33 = v32 + 1;
                    if ( v31 == sub_15F4DF0(v95, v32 + 1)
                      || (v33 = v32 + 2, v31 == sub_15F4DF0(v95, v32 + 2))
                      || (v33 = v32 + 3, v31 == sub_15F4DF0(v95, v32 + 3)) )
                    {
                      v32 = v33;
                      goto LABEL_45;
                    }
                    v32 += 4;
                    if ( v32 == 4 * (v107 >> 2) )
                    {
                      v53 = v107 - v32;
                      goto LABEL_76;
                    }
                  }
                  goto LABEL_45;
                }
                v53 = v107;
                v32 = 0;
LABEL_76:
                if ( v53 == 2 )
                  goto LABEL_90;
                if ( v53 == 3 )
                  break;
                if ( v53 != 1 )
                  goto LABEL_79;
LABEL_92:
                if ( v31 != sub_15F4DF0(v95, v32) )
                {
LABEL_79:
                  v18 = v101;
                  v41 = (unsigned __int64)v117;
                  goto LABEL_63;
                }
LABEL_45:
                if ( v107 == v32 )
                  goto LABEL_79;
                v111 += 24;
                if ( v105 == v111 )
                {
                  v18 = v101;
                  goto LABEL_48;
                }
              }
              if ( v31 == sub_15F4DF0(v95, v32) )
                goto LABEL_45;
              ++v32;
LABEL_90:
              if ( v31 == sub_15F4DF0(v95, v32) )
                goto LABEL_45;
              ++v32;
              goto LABEL_92;
            }
LABEL_48:
            v34 = *(unsigned int *)(a4 + 8);
            v115 = 0x400000000LL;
            v35 = *(unsigned int *)(a4 + 12);
            v113 = v102;
            v114 = v116;
            if ( (unsigned int)v34 >= (unsigned int)v35 )
            {
              v69 = ((((unsigned __int64)(v35 + 2) >> 1) | (v35 + 2)) >> 2)
                  | ((unsigned __int64)(v35 + 2) >> 1)
                  | (v35 + 2);
              v70 = (v69 >> 4) | v69;
              v71 = ((v70 >> 8) | v70 | (((v70 >> 8) | v70) >> 16) | (((v70 >> 8) | v70) >> 32)) + 1;
              v72 = 0xFFFFFFFFLL;
              if ( v71 <= 0xFFFFFFFF )
                v72 = v71;
              v112 = v72;
              v36 = malloc(56 * v72);
              if ( !v36 )
              {
                sub_16BD1C0("Allocation failed", 1u);
                v34 = *(unsigned int *)(a4 + 8);
              }
              v73 = *(_QWORD *)a4;
              v29 = *(_QWORD *)a4 + 56 * v34;
              if ( *(_QWORD *)a4 != v29 )
              {
                v74 = *(_QWORD *)a4;
                v75 = v29;
                v108 = v36;
                do
                {
                  while ( 1 )
                  {
                    if ( v36 )
                    {
                      v76 = *(_QWORD *)v74;
                      *(_DWORD *)(v36 + 16) = 0;
                      *(_DWORD *)(v36 + 20) = 4;
                      *(_QWORD *)v36 = v76;
                      *(_QWORD *)(v36 + 8) = v36 + 24;
                      v77 = *(unsigned int *)(v74 + 16);
                      if ( (_DWORD)v77 )
                        break;
                    }
                    v74 += 56;
                    v36 += 56;
                    if ( v75 == v74 )
                      goto LABEL_128;
                  }
                  v78 = (char **)(v74 + 8);
                  v79 = v36 + 8;
                  v74 += 56;
                  v36 += 56;
                  sub_191FDF0(v79, v78, v73, v77, v29, v30);
                }
                while ( v75 != v74 );
LABEL_128:
                v36 = v108;
                v29 = *(_QWORD *)a4;
                v80 = *(_QWORD *)a4 + 56LL * *(unsigned int *)(a4 + 8);
                if ( v80 != *(_QWORD *)a4 )
                {
                  v81 = *(_QWORD *)a4;
                  do
                  {
                    v80 -= 56;
                    v82 = *(_QWORD *)(v80 + 8);
                    if ( v82 != v80 + 24 )
                      _libc_free(v82);
                  }
                  while ( v80 != v81 );
                  v36 = v108;
                  v29 = *(_QWORD *)a4;
                }
              }
              if ( v29 != a4 + 16 )
                _libc_free(v29);
              *(_QWORD *)a4 = v36;
              LODWORD(v34) = *(_DWORD *)(a4 + 8);
              *(_DWORD *)(a4 + 12) = v112;
            }
            else
            {
              v36 = *(_QWORD *)a4;
            }
            v37 = (__int64 *)(v36 + 56LL * (unsigned int)v34);
            if ( v37 )
            {
              *v37 = v113;
              v37[1] = (__int64)(v37 + 3);
              v37[2] = 0x400000000LL;
              if ( (_DWORD)v115 )
                sub_191FDF0((__int64)(v37 + 1), &v114, (unsigned int)v115, (unsigned int)v34, v29, v30);
              LODWORD(v34) = *(_DWORD *)(a4 + 8);
            }
            v38 = v114;
            v39 = (unsigned int)(v34 + 1);
            *(_DWORD *)(a4 + 8) = v39;
            if ( v38 != v116 )
            {
              _libc_free((unsigned __int64)v38);
              v39 = *(unsigned int *)(a4 + 8);
            }
            v40 = 7 * v39;
            v41 = (unsigned __int64)v117;
            v42 = *(_QWORD *)a4 + 8 * v40 - 56;
            v43 = &v117[3 * (unsigned int)v118];
            if ( v43 != v117 )
            {
              v44 = *(unsigned int *)(v42 + 16);
              v45 = &v117[3 * (unsigned int)v118];
              do
              {
                v46 = *(_QWORD *)(v41 + 16);
                if ( *(_DWORD *)(v42 + 20) <= (unsigned int)v44 )
                {
                  sub_16CD150(v42 + 8, (const void *)(v42 + 24), 0, 8, (int)v43, v30);
                  v44 = *(unsigned int *)(v42 + 16);
                }
                v41 += 24LL;
                *(_QWORD *)(*(_QWORD *)(v42 + 8) + 8 * v44) = v46;
                v44 = (unsigned int)(*(_DWORD *)(v42 + 16) + 1);
                *(_DWORD *)(v42 + 16) = v44;
              }
              while ( (char **)v41 != v45 );
              v17 = v95;
LABEL_62:
              v41 = (unsigned __int64)v117;
            }
LABEL_63:
            v47 = v93[1] + 24LL * *((unsigned int *)v93 + 4);
            v48 = (__int64)(0xAAAAAAAAAAAAAAABLL * ((v47 - (__int64)v109) >> 3)) >> 2;
            if ( v48 > 0 )
            {
              v49 = *(_DWORD *)v109;
              v50 = (char *)v18;
              v51 = (char *)(v18 + 96 * v48);
              while ( *(_DWORD *)v50 == v49 )
              {
                v65 = *((_DWORD *)v109 + 1);
                if ( *((_DWORD *)v50 + 1) != v65 )
                  break;
                v66 = v50 + 24;
                if ( v49 != *((_DWORD *)v50 + 6)
                  || v65 != *((_DWORD *)v50 + 7)
                  || (v66 = v50 + 48, v49 != *((_DWORD *)v50 + 12))
                  || v65 != *((_DWORD *)v50 + 13)
                  || (v66 = v50 + 72, v49 != *((_DWORD *)v50 + 18))
                  || v65 != *((_DWORD *)v50 + 19) )
                {
                  v50 = v66;
                  goto LABEL_66;
                }
                v50 += 96;
                if ( v51 == v50 )
                  goto LABEL_114;
              }
              goto LABEL_66;
            }
            v50 = (char *)v18;
LABEL_114:
            v68 = v47 - (_QWORD)v50;
            if ( v47 - (_QWORD)v50 == 48 )
            {
              v87 = *(_DWORD *)v109;
              goto LABEL_163;
            }
            if ( v68 == 72 )
            {
              v87 = *(_DWORD *)v109;
              if ( *(_DWORD *)v50 != *(_DWORD *)v109 || *((_DWORD *)v50 + 1) != *((_DWORD *)v109 + 1) )
                goto LABEL_66;
              v50 += 24;
LABEL_163:
              if ( *(_DWORD *)v50 != v87 || *((_DWORD *)v50 + 1) != *((_DWORD *)v109 + 1) )
                goto LABEL_66;
              v50 += 24;
LABEL_155:
              if ( *(_DWORD *)v50 == v87 && *((_DWORD *)v50 + 1) == *((_DWORD *)v109 + 1) )
                v50 = (char *)(v93[1] + 24LL * *((unsigned int *)v93 + 4));
              goto LABEL_66;
            }
            if ( v68 == 24 )
            {
              v87 = *(_DWORD *)v109;
              goto LABEL_155;
            }
            v50 = (char *)(v93[1] + 24LL * *((unsigned int *)v93 + 4));
LABEL_66:
            if ( (char **)v41 != v119 )
              _libc_free(v41);
            if ( v109 != v50 )
            {
              v109 = v50;
              continue;
            }
            break;
          }
LABEL_98:
          v94 = v93 + 9;
          if ( v94 == v90 )
            return;
          v67 = v94;
          while ( *v67 == -16 || *v67 == -8 )
          {
            v67 += 9;
            if ( v90 == v67 )
              return;
          }
          v93 = v67;
          if ( v90 == v67 )
            return;
          v102 = *v67;
        }
        v89 = *(_DWORD *)v9;
LABEL_174:
        if ( *(_DWORD *)v109 != v89 || *((_DWORD *)v109 + 1) != *((_DWORD *)v9 + 1) )
          goto LABEL_15;
        v109 += 24;
        goto LABEL_177;
      }
    }
  }
}
