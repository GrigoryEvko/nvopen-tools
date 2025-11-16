// Function: sub_FFC290
// Address: 0xffc290
//
void __fastcall sub_FFC290(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ebx
  __int64 *v7; // r14
  __int64 *v8; // r13
  char v9; // di
  __int64 v10; // rsi
  char *v11; // rax
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 *v15; // r15
  __int64 v16; // r10
  __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rdx
  _QWORD *v23; // r8
  __int64 v24; // r12
  char *v25; // rax
  char *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r14
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // r12
  unsigned int v33; // eax
  __int64 v34; // rax
  __int64 *v35; // r12
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rcx
  __int64 v40; // r13
  _QWORD *v41; // rax
  __int64 v42; // r9
  __int64 v43; // r8
  int v44; // eax
  __int64 v45; // rax
  __int64 v46; // r10
  unsigned int v47; // ecx
  __int64 v48; // rcx
  __int64 v49; // r14
  __int64 v50; // rdi
  __int64 v51; // rax
  unsigned __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rcx
  unsigned int v56; // eax
  __int64 v57; // r14
  __int64 v58; // r13
  _QWORD *v59; // rdi
  _QWORD *v60; // rax
  __int64 v61; // r8
  int v62; // r9d
  size_t v63; // rdx
  __int64 v64; // r9
  __int64 v65; // rax
  _QWORD *v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 *v70; // r14
  __int64 *v71; // rbx
  __int64 v72; // r15
  _QWORD *v73; // rax
  unsigned __int64 v74; // r12
  __int64 *v75; // rax
  __int64 v76; // rsi
  unsigned int v77; // ebx
  __int64 v78; // rax
  __int64 v79; // r14
  _QWORD *v80; // rax
  _QWORD *v81; // r14
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rdi
  int v85; // r14d
  unsigned int v86; // eax
  __int64 v87; // rbx
  __int64 v88; // rdx
  __int64 v89; // rax
  _QWORD *v90; // rdi
  int v91; // r12d
  _QWORD *v92; // rcx
  __int64 *v93; // rdx
  _QWORD *v94; // rsi
  __int64 v95; // rax
  __int64 v96; // r14
  __int64 v97; // rbx
  __int64 v98; // r8
  __int64 v99; // rdi
  __int64 v100; // [rsp+8h] [rbp-3C8h]
  _QWORD *v101; // [rsp+10h] [rbp-3C0h]
  __int64 v102; // [rsp+18h] [rbp-3B8h]
  __int64 v103; // [rsp+20h] [rbp-3B0h]
  __int64 v104; // [rsp+28h] [rbp-3A8h]
  __int64 v106; // [rsp+30h] [rbp-3A0h]
  unsigned int v107; // [rsp+38h] [rbp-398h]
  __int64 v108; // [rsp+38h] [rbp-398h]
  unsigned int v109; // [rsp+38h] [rbp-398h]
  __int64 *v111; // [rsp+48h] [rbp-388h]
  __int64 v112; // [rsp+50h] [rbp-380h]
  __int64 v113; // [rsp+50h] [rbp-380h]
  __int64 v114; // [rsp+50h] [rbp-380h]
  _QWORD *v115; // [rsp+50h] [rbp-380h]
  _QWORD *v116; // [rsp+50h] [rbp-380h]
  __int64 v117; // [rsp+50h] [rbp-380h]
  __int64 v118; // [rsp+50h] [rbp-380h]
  __int64 v119; // [rsp+50h] [rbp-380h]
  __int64 v120; // [rsp+50h] [rbp-380h]
  __int64 v121; // [rsp+50h] [rbp-380h]
  unsigned int v122; // [rsp+58h] [rbp-378h]
  __int64 v123; // [rsp+58h] [rbp-378h]
  __int64 v124; // [rsp+58h] [rbp-378h]
  unsigned __int64 v125; // [rsp+68h] [rbp-368h] BYREF
  __int64 v126; // [rsp+70h] [rbp-360h] BYREF
  char *v127; // [rsp+78h] [rbp-358h]
  __int64 v128; // [rsp+80h] [rbp-350h]
  int v129; // [rsp+88h] [rbp-348h]
  char v130; // [rsp+8Ch] [rbp-344h]
  char v131; // [rsp+90h] [rbp-340h] BYREF
  _QWORD *v132; // [rsp+190h] [rbp-240h] BYREF
  __int64 v133; // [rsp+198h] [rbp-238h]
  _QWORD v134[70]; // [rsp+1A0h] [rbp-230h] BYREF

  if ( *(_QWORD *)(a1 + 544) )
  {
    v6 = a3;
    if ( a3 )
    {
      v126 = 0;
      v127 = &v131;
      v7 = &a2[3 * a3];
      v130 = 1;
      v128 = 32;
      v129 = 0;
      v111 = v7;
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
LABEL_113:
              v8 += 3;
              sub_C8CC70((__int64)&v126, v10, a3, a4, a5, a6);
              v9 = v130;
              if ( v8 == v7 )
                goto LABEL_11;
            }
            v11 = v127;
            a4 = HIDWORD(v128);
            a3 = (__int64)&v127[8 * HIDWORD(v128)];
            if ( v127 != (char *)a3 )
              break;
LABEL_115:
            if ( HIDWORD(v128) >= (unsigned int)v128 )
              goto LABEL_113;
            a4 = (unsigned int)(HIDWORD(v128) + 1);
            v8 += 3;
            ++HIDWORD(v128);
            *(_QWORD *)a3 = v10;
            v9 = v130;
            ++v126;
            if ( v8 == v7 )
              goto LABEL_11;
          }
          while ( v10 != *(_QWORD *)v11 )
          {
            v11 += 8;
            if ( (char *)a3 == v11 )
              goto LABEL_115;
          }
          v8 += 3;
        }
        while ( v8 != v7 );
LABEL_11:
        sub_B48880((__int64 *)&v125, v6, 1u);
        v107 = 0;
        v13 = *(unsigned int *)(*(_QWORD *)(a1 + 544) + 32LL);
        v14 = *(_QWORD *)(a1 + 544);
        v15 = a2;
        v16 = a1;
        while ( 1 )
        {
          v17 = v15[1];
          if ( v17 )
          {
            v18 = (unsigned int)(*(_DWORD *)(v17 + 44) + 1);
            v19 = *(_DWORD *)(v17 + 44) + 1;
          }
          else
          {
            v18 = 0;
            v19 = 0;
          }
          v20 = 0;
          if ( v19 < (unsigned int)v13 )
            v20 = *(_QWORD *)(*(_QWORD *)(v14 + 24) + 8 * v18);
          v21 = *(_QWORD *)(v17 + 16);
          if ( !v21 )
            goto LABEL_46;
          while ( 1 )
          {
            v22 = *(_QWORD *)(v21 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
              break;
            v21 = *(_QWORD *)(v21 + 8);
            if ( !v21 )
              goto LABEL_46;
          }
          v23 = v134;
LABEL_19:
          v24 = *(_QWORD *)(v22 + 40);
          if ( v15[2] == v24 )
            goto LABEL_102;
          if ( v130 )
            break;
          v116 = v23;
          v123 = v16;
          v75 = sub_C8CA60((__int64)&v126, v24);
          v16 = v123;
          v23 = v116;
          if ( v75 )
            goto LABEL_25;
          v29 = *(_QWORD *)(v123 + 544);
          if ( !v24 )
          {
LABEL_111:
            v30 = 0;
            v31 = 0;
            goto LABEL_30;
          }
LABEL_29:
          v30 = (unsigned int)(*(_DWORD *)(v24 + 44) + 1);
          v31 = *(_DWORD *)(v24 + 44) + 1;
LABEL_30:
          if ( v31 >= *(_DWORD *)(v29 + 32) )
            goto LABEL_102;
          v32 = *(_QWORD *)(*(_QWORD *)(v29 + 24) + 8 * v30);
          if ( v20 == v32 || !v32 )
            goto LABEL_102;
          if ( !v20 )
            goto LABEL_43;
          if ( v20 == *(_QWORD *)(v32 + 8) )
            goto LABEL_102;
          if ( v32 == *(_QWORD *)(v20 + 8) || *(_DWORD *)(v20 + 16) >= *(_DWORD *)(v32 + 16) )
            goto LABEL_43;
          if ( *(_BYTE *)(v29 + 112) )
            goto LABEL_100;
          v33 = *(_DWORD *)(v29 + 116) + 1;
          *(_DWORD *)(v29 + 116) = v33;
          if ( v33 > 0x20 )
          {
            v132 = v23;
            HIDWORD(v133) = 32;
            v82 = *(_QWORD *)(v29 + 96);
            if ( v82 )
            {
              v83 = *(_QWORD *)(v82 + 24);
              v134[0] = *(_QWORD *)(v29 + 96);
              v84 = (__int64)v23;
              LODWORD(v133) = 1;
              v134[1] = v83;
              v124 = v29;
              v85 = 1;
              *(_DWORD *)(v82 + 72) = 0;
              v86 = 1;
              v117 = v32;
              v103 = v21;
              v102 = v16;
              do
              {
                v91 = v85++;
                v92 = (_QWORD *)(v84 + 16LL * v86 - 16);
                v94 = (_QWORD *)*v92;
                v93 = (__int64 *)v92[1];
                if ( v93 == (__int64 *)(*(_QWORD *)(*v92 + 24LL) + 8LL * *(unsigned int *)(*v92 + 32LL)) )
                {
                  --v86;
                  *((_DWORD *)v94 + 19) = v91;
                  LODWORD(v133) = v86;
                }
                else
                {
                  v87 = *v93;
                  v92[1] = v93 + 1;
                  v88 = (unsigned int)v133;
                  v89 = *(_QWORD *)(v87 + 24);
                  if ( (unsigned __int64)(unsigned int)v133 + 1 > HIDWORD(v133) )
                  {
                    v94 = v23;
                    v100 = *(_QWORD *)(v87 + 24);
                    v101 = v23;
                    sub_C8D5F0((__int64)&v132, v23, (unsigned int)v133 + 1LL, 0x10u, (__int64)v23, v12);
                    v84 = (__int64)v132;
                    v88 = (unsigned int)v133;
                    v89 = v100;
                    v23 = v101;
                  }
                  v90 = (_QWORD *)(16 * v88 + v84);
                  *v90 = v87;
                  v90[1] = v89;
                  v86 = v133 + 1;
                  LODWORD(v133) = v133 + 1;
                  *(_DWORD *)(v87 + 72) = v91;
                  v84 = (__int64)v132;
                }
              }
              while ( v86 );
              v32 = v117;
              v21 = v103;
              v16 = v102;
              *(_DWORD *)(v124 + 116) = 0;
              *(_BYTE *)(v124 + 112) = 1;
              if ( (_QWORD *)v84 != v23 )
              {
                v115 = v23;
                _libc_free(v84, v94);
                v16 = v102;
                v23 = v115;
              }
            }
LABEL_100:
            if ( *(_DWORD *)(v32 + 72) >= *(_DWORD *)(v20 + 72) && *(_DWORD *)(v32 + 76) <= *(_DWORD *)(v20 + 76) )
              goto LABEL_102;
            goto LABEL_43;
          }
          do
          {
            v34 = v32;
            v32 = *(_QWORD *)(v32 + 8);
          }
          while ( v32 && *(_DWORD *)(v20 + 16) <= *(_DWORD *)(v32 + 16) );
          if ( v20 == v34 )
          {
LABEL_102:
            while ( 1 )
            {
              v21 = *(_QWORD *)(v21 + 8);
              if ( !v21 )
                goto LABEL_45;
              v22 = *(_QWORD *)(v21 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
                goto LABEL_19;
            }
          }
LABEL_43:
          if ( (v125 & 1) != 0 )
            v125 = 2 * ((v125 >> 58 << 57) | ~(-1LL << (v125 >> 58)) & (v125 >> 1) & ~(1LL << v107)) + 1;
          else
            *(_QWORD *)(*(_QWORD *)v125 + 8LL * (v107 >> 6)) &= ~(1LL << v107);
LABEL_45:
          v14 = *(_QWORD *)(v16 + 544);
          v13 = *(unsigned int *)(v14 + 32);
LABEL_46:
          ++v107;
          v15 += 3;
          if ( v111 == v15 )
          {
            v35 = a2;
            v36 = v14;
            v122 = 0;
            v37 = *a2;
            v38 = a2[2];
            if ( !*a2 )
              goto LABEL_90;
LABEL_48:
            v39 = (unsigned int)(*(_DWORD *)(v37 + 44) + 1);
            if ( *(_DWORD *)(v37 + 44) + 1 >= (unsigned int)v13 )
            {
              while ( 1 )
              {
                *(_BYTE *)(v36 + 112) = 0;
                v73 = (_QWORD *)sub_22077B0(80);
                v43 = (__int64)v73;
                if ( v73 )
                  break;
                v40 = 0;
LABEL_53:
                if ( v38 )
                {
                  v45 = (unsigned int)(*(_DWORD *)(v38 + 44) + 1);
                  v46 = 8 * v45;
                }
                else
                {
                  v46 = 0;
                  LODWORD(v45) = 0;
                }
                v47 = *(_DWORD *)(v36 + 32);
                if ( v47 > (unsigned int)v45 )
                  goto LABEL_56;
                v76 = *(_QWORD *)(v36 + 104);
                v77 = v45 + 1;
                v78 = v47;
                if ( *(_DWORD *)(v76 + 88) >= v77 )
                  v77 = *(_DWORD *)(v76 + 88);
                v13 = v77;
                if ( v77 == (unsigned __int64)v47 )
                {
LABEL_56:
                  v48 = *(_QWORD *)(v36 + 24);
                }
                else
                {
                  v79 = 8LL * v77;
                  if ( v77 < (unsigned __int64)v47 )
                  {
                    v48 = *(_QWORD *)(v36 + 24);
                    v95 = v48 + 8 * v78;
                    v96 = v48 + v79;
                    if ( v95 != v96 )
                    {
                      v119 = v43;
                      v106 = v46;
                      v109 = v77;
                      v97 = v95;
                      do
                      {
                        v98 = *(_QWORD *)(v97 - 8);
                        v97 -= 8;
                        if ( v98 )
                        {
                          v99 = *(_QWORD *)(v98 + 24);
                          if ( v99 != v98 + 40 )
                          {
                            v104 = v98;
                            _libc_free(v99, v13);
                            v98 = v104;
                          }
                          v13 = 80;
                          j_j___libc_free_0(v98, 80);
                        }
                      }
                      while ( v96 != v97 );
                      v43 = v119;
                      v77 = v109;
                      v46 = v106;
                      v48 = *(_QWORD *)(v36 + 24);
                    }
                  }
                  else
                  {
                    if ( v77 > (unsigned __int64)*(unsigned int *)(v36 + 36) )
                    {
                      v108 = v46;
                      v118 = v43;
                      sub_B1B4E0(v36 + 24, v77);
                      v78 = *(unsigned int *)(v36 + 32);
                      v46 = v108;
                      v43 = v118;
                    }
                    v48 = *(_QWORD *)(v36 + 24);
                    v80 = (_QWORD *)(v48 + 8 * v78);
                    v81 = (_QWORD *)(v48 + v79);
                    if ( v80 != v81 )
                    {
                      do
                      {
                        if ( v80 )
                          *v80 = 0;
                        ++v80;
                      }
                      while ( v81 != v80 );
                      v48 = *(_QWORD *)(v36 + 24);
                    }
                  }
                  *(_DWORD *)(v36 + 32) = v77;
                }
                v49 = *(_QWORD *)(v48 + v46);
                *(_QWORD *)(v48 + v46) = v43;
                if ( v49 )
                {
                  v50 = *(_QWORD *)(v49 + 24);
                  if ( v50 != v49 + 40 )
                  {
                    v112 = v43;
                    _libc_free(v50, v13);
                    v43 = v112;
                  }
                  v13 = 80;
                  v113 = v43;
                  j_j___libc_free_0(v49, 80);
                  v43 = v113;
                }
                if ( v40 )
                {
                  v51 = *(unsigned int *)(v40 + 32);
                  if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 36) )
                  {
                    v13 = v40 + 40;
                    v120 = v43;
                    sub_C8D5F0(v40 + 24, (const void *)(v40 + 40), v51 + 1, 8u, v43, v42);
                    v51 = *(unsigned int *)(v40 + 32);
                    v43 = v120;
                  }
                  *(_QWORD *)(*(_QWORD *)(v40 + 24) + 8 * v51) = v43;
                  ++*(_DWORD *)(v40 + 32);
                }
                if ( (v125 & 1) != 0 )
                {
                  v52 = (((v125 >> 1) & ~(-1LL << (v125 >> 58))) >> v122) & 1;
                }
                else
                {
                  v13 = v122;
                  v52 = (*(_QWORD *)(*(_QWORD *)v125 + 8LL * (v122 >> 6)) >> v122) & 1LL;
                }
                if ( (_BYTE)v52 )
                {
                  v53 = *(_QWORD *)(a1 + 544);
                  v54 = v35[1];
                  if ( v54 )
                  {
                    v55 = (unsigned int)(*(_DWORD *)(v54 + 44) + 1);
                    v56 = *(_DWORD *)(v54 + 44) + 1;
                  }
                  else
                  {
                    v55 = 0;
                    v56 = 0;
                  }
                  if ( v56 >= *(_DWORD *)(v53 + 32) )
                  {
                    *(_BYTE *)(v53 + 112) = 0;
                    BUG();
                  }
                  v57 = *(_QWORD *)(*(_QWORD *)(v53 + 24) + 8 * v55);
                  *(_BYTE *)(v53 + 112) = 0;
                  v58 = *(_QWORD *)(v57 + 8);
                  if ( v43 != v58 )
                  {
                    v132 = (_QWORD *)v57;
                    v59 = *(_QWORD **)(v58 + 24);
                    v13 = (__int64)&v59[*(unsigned int *)(v58 + 32)];
                    v60 = sub_FFB140(v59, v13, (__int64 *)&v132);
                    if ( v60 + 1 != (_QWORD *)v13 )
                    {
                      v114 = v61;
                      v63 = v13 - (_QWORD)(v60 + 1);
                      v13 = (__int64)(v60 + 1);
                      memmove(v60, v60 + 1, v63);
                      v62 = *(_DWORD *)(v58 + 32);
                      v61 = v114;
                    }
                    v64 = (unsigned int)(v62 - 1);
                    *(_DWORD *)(v58 + 32) = v64;
                    *(_QWORD *)(v57 + 8) = v61;
                    v65 = *(unsigned int *)(v61 + 32);
                    if ( v65 + 1 > (unsigned __int64)*(unsigned int *)(v61 + 36) )
                    {
                      v13 = v61 + 40;
                      v121 = v61;
                      sub_C8D5F0(v61 + 24, (const void *)(v61 + 40), v65 + 1, 8u, v61, v64);
                      v61 = v121;
                      v65 = *(unsigned int *)(v121 + 32);
                    }
                    *(_QWORD *)(*(_QWORD *)(v61 + 24) + 8 * v65) = v57;
                    ++*(_DWORD *)(v61 + 32);
                    if ( *(_DWORD *)(v57 + 16) != *(_DWORD *)(*(_QWORD *)(v57 + 8) + 16LL) + 1 )
                    {
                      v134[0] = v57;
                      v132 = v134;
                      v66 = v134;
                      v133 = 0x4000000001LL;
                      LODWORD(v67) = 1;
                      do
                      {
                        v68 = (unsigned int)v67;
                        v67 = (unsigned int)(v67 - 1);
                        v69 = v66[v68 - 1];
                        LODWORD(v133) = v67;
                        v70 = *(__int64 **)(v69 + 24);
                        *(_DWORD *)(v69 + 16) = *(_DWORD *)(*(_QWORD *)(v69 + 8) + 16LL) + 1;
                        v71 = &v70[*(unsigned int *)(v69 + 32)];
                        if ( v70 != v71 )
                        {
                          do
                          {
                            v72 = *v70;
                            if ( *(_DWORD *)(*v70 + 16) != *(_DWORD *)(*(_QWORD *)(*v70 + 8) + 16LL) + 1 )
                            {
                              if ( v67 + 1 > (unsigned __int64)HIDWORD(v133) )
                              {
                                v13 = (__int64)v134;
                                sub_C8D5F0((__int64)&v132, v134, v67 + 1, 8u, v61, v64);
                                v67 = (unsigned int)v133;
                              }
                              v132[v67] = v72;
                              v67 = (unsigned int)(v133 + 1);
                              LODWORD(v133) = v133 + 1;
                            }
                            ++v70;
                          }
                          while ( v71 != v70 );
                          v66 = v132;
                        }
                      }
                      while ( (_DWORD)v67 );
                      if ( v66 != v134 )
                        _libc_free(v66, v13);
                    }
                  }
                }
                ++v122;
                v35 += 3;
                if ( v111 == v35 )
                  goto LABEL_94;
                v38 = v35[2];
                v36 = *(_QWORD *)(a1 + 544);
                v37 = *v35;
                v13 = *(unsigned int *)(v36 + 32);
                if ( *v35 )
                  goto LABEL_48;
LABEL_90:
                v39 = 0;
                if ( (_DWORD)v13 )
                  goto LABEL_49;
              }
              *v73 = v38;
              v40 = 0;
              v44 = 0;
              *(_QWORD *)(v43 + 8) = 0;
            }
            else
            {
LABEL_49:
              v40 = *(_QWORD *)(*(_QWORD *)(v36 + 24) + 8 * v39);
              *(_BYTE *)(v36 + 112) = 0;
              v41 = (_QWORD *)sub_22077B0(80);
              v43 = (__int64)v41;
              if ( !v41 )
                goto LABEL_53;
              *v41 = v38;
              v41[1] = v40;
              if ( v40 )
                v44 = *(_DWORD *)(v40 + 16) + 1;
              else
                v44 = 0;
            }
            *(_DWORD *)(v43 + 16) = v44;
            *(_QWORD *)(v43 + 24) = v43 + 40;
            *(_QWORD *)(v43 + 32) = 0x400000000LL;
            *(_QWORD *)(v43 + 72) = -1;
            goto LABEL_53;
          }
        }
        v25 = v127;
        v26 = &v127[8 * HIDWORD(v128)];
        if ( v127 != v26 )
        {
          while ( v24 != *(_QWORD *)v25 )
          {
            v25 += 8;
            if ( v26 == v25 )
              goto LABEL_28;
          }
LABEL_25:
          v27 = *(_QWORD *)(v24 + 16);
          if ( !v27 )
LABEL_106:
            BUG();
          while ( 1 )
          {
            v28 = *(_QWORD *)(v27 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v28 - 30) <= 0xAu )
              break;
            v27 = *(_QWORD *)(v27 + 8);
            if ( !v27 )
              goto LABEL_106;
          }
          v24 = *(_QWORD *)(v28 + 40);
        }
LABEL_28:
        v29 = *(_QWORD *)(v16 + 544);
        if ( !v24 )
          goto LABEL_111;
        goto LABEL_29;
      }
      v13 = (unsigned int)a3;
      sub_B48880((__int64 *)&v125, a3, 1u);
LABEL_94:
      v74 = v125;
      if ( (v125 & 1) != 0 || !v125 )
      {
        if ( v130 )
          return;
LABEL_150:
        _libc_free(v127, v13);
        return;
      }
      if ( *(_QWORD *)v125 != v125 + 16 )
        _libc_free(*(_QWORD *)v125, v13);
      v13 = 72;
      j_j___libc_free_0(v74, 72);
      if ( !v130 )
        goto LABEL_150;
    }
  }
}
