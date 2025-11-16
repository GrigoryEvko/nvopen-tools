// Function: sub_D2DA90
// Address: 0xd2da90
//
void __fastcall sub_D2DA90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v5; // r12
  unsigned int i; // r8d
  int v7; // r10d
  __int64 *v8; // rdi
  __int64 v9; // rdx
  __int64 *v10; // rbx
  _QWORD *v11; // rsi
  __int64 v12; // rdx
  _BYTE *v13; // r12
  __int64 v14; // r8
  int v15; // ecx
  __int64 v16; // rsi
  __int64 v17; // rdi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  __int64 v22; // rsi
  __int64 v23; // r13
  _QWORD *v24; // rax
  int v25; // esi
  int v26; // edx
  int v27; // eax
  __int64 v28; // rax
  _QWORD *v29; // r12
  _QWORD *v30; // rbx
  _QWORD *v31; // rdi
  __int64 *v32; // r13
  int v33; // esi
  __int64 v34; // r14
  __int64 v35; // rdi
  int v36; // esi
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // r8
  __int64 v40; // rbx
  unsigned int v41; // esi
  __int64 v42; // rdi
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // r10
  __int64 v46; // rdi
  unsigned int v47; // edx
  __int64 *v48; // rax
  __int64 v49; // r10
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rdi
  int v53; // eax
  int v54; // eax
  int v55; // eax
  __int64 v56; // r9
  _QWORD *v57; // rax
  __int64 v58; // rdx
  unsigned int v59; // ebx
  __int64 *v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // rsi
  __int64 *v63; // r12
  char *v64; // rbx
  int v65; // eax
  __int64 v66; // rcx
  int v67; // edi
  unsigned int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // r8
  __int64 *v71; // rax
  __int64 v72; // r13
  _QWORD *v73; // rax
  int v74; // eax
  int v75; // r9d
  char *v76; // r14
  size_t v77; // rdx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // r14
  __int64 v81; // r12
  _QWORD *v82; // rdx
  _QWORD *v83; // r13
  __int64 v84; // rax
  _QWORD *v85; // rbx
  int v86; // ecx
  unsigned __int64 v87; // rax
  __int64 v88; // rsi
  __int64 v89; // r15
  int v90; // ecx
  unsigned int v91; // edx
  __int64 *v92; // rax
  __int64 v93; // rdi
  _QWORD *v94; // rax
  _QWORD *v95; // rdx
  __int64 v96; // rax
  unsigned __int64 v97; // rdx
  __int64 *v98; // rax
  int v99; // eax
  __int64 v100; // rdi
  int v101; // r9d
  int v102; // r9d
  int v103; // r9d
  int v104; // r9d
  __int64 *v105; // [rsp+10h] [rbp-120h]
  __int64 *v106; // [rsp+28h] [rbp-108h]
  _QWORD *v107; // [rsp+38h] [rbp-F8h]
  _QWORD *v108; // [rsp+48h] [rbp-E8h]
  __int64 *v109; // [rsp+50h] [rbp-E0h]
  unsigned int v110; // [rsp+58h] [rbp-D8h]
  __int64 *v111; // [rsp+58h] [rbp-D8h]
  __int64 **v112; // [rsp+60h] [rbp-D0h] BYREF
  char v113; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v114; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v115; // [rsp+88h] [rbp-A8h]
  __int64 v116; // [rsp+90h] [rbp-A0h]
  unsigned int v117; // [rsp+98h] [rbp-98h]
  _QWORD *v118; // [rsp+A0h] [rbp-90h] BYREF
  char *v119; // [rsp+A8h] [rbp-88h] BYREF
  __int64 v120; // [rsp+B0h] [rbp-80h]
  char v121[8]; // [rsp+B8h] [rbp-78h] BYREF
  __int64 *v122; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v123; // [rsp+C8h] [rbp-68h]
  _BYTE v124[96]; // [rsp+D0h] [rbp-60h] BYREF

  v105 = (__int64 *)a2;
  if ( a3 )
  {
    v3 = (__int64 *)a2;
    v114 = 0;
    v115 = 0;
    v116 = 0;
    v117 = 0;
    v106 = (__int64 *)(a2 + 8 * a3);
    if ( (__int64 *)a2 != v106 )
    {
      v5 = 0;
      for ( i = 0; ; i = v117 )
      {
        v15 = *(_DWORD *)(a1 + 120);
        v16 = *v3;
        v17 = *(_QWORD *)(a1 + 104);
        if ( v15 )
        {
          v18 = v15 - 1;
          v19 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( v16 == *v20 )
          {
LABEL_11:
            v22 = v20[1];
            v23 = v22;
            goto LABEL_12;
          }
          v27 = 1;
          while ( v21 != -4096 )
          {
            v103 = v27 + 1;
            v19 = v18 & (v27 + v19);
            v20 = (__int64 *)(v17 + 16LL * v19);
            v21 = *v20;
            if ( v16 == *v20 )
              goto LABEL_11;
            v27 = v103;
          }
        }
        v23 = 0;
        v22 = 0;
LABEL_12:
        v110 = i;
        v24 = (_QWORD *)sub_D23C40(a1, v22);
        v14 = v110;
        if ( v24 )
          v24 = (_QWORD *)*v24;
        v118 = v24;
        if ( !v110 )
        {
          ++v114;
          v122 = 0;
          goto LABEL_16;
        }
        v7 = 1;
        v8 = 0;
        LODWORD(v9) = (v110 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v10 = (__int64 *)(v5 + 32LL * (unsigned int)v9);
        v11 = (_QWORD *)*v10;
        if ( v24 != (_QWORD *)*v10 )
        {
          while ( v11 != (_QWORD *)-4096LL )
          {
            if ( !v8 && v11 == (_QWORD *)-8192LL )
              v8 = v10;
            v56 = (unsigned int)(v7 + 1);
            v9 = (v110 - 1) & ((_DWORD)v9 + v7);
            v10 = (__int64 *)(v5 + 32 * v9);
            v11 = (_QWORD *)*v10;
            if ( v24 == (_QWORD *)*v10 )
              goto LABEL_5;
            ++v7;
          }
          if ( v8 )
            v10 = v8;
          ++v114;
          v26 = v116 + 1;
          v122 = v10;
          if ( 4 * ((int)v116 + 1) >= 3 * v110 )
          {
LABEL_16:
            v25 = 2 * v110;
          }
          else
          {
            if ( v110 - (v26 + HIDWORD(v116)) > v110 >> 3 )
              goto LABEL_62;
            v25 = v110;
          }
          sub_D277E0((__int64)&v114, v25);
          sub_D24DD0((__int64)&v114, (__int64 *)&v118, &v122);
          v24 = v118;
          v10 = v122;
          v26 = v116 + 1;
LABEL_62:
          LODWORD(v116) = v26;
          if ( *v10 != -4096 )
            --HIDWORD(v116);
          *v10 = (__int64)v24;
          v13 = v10 + 1;
          v12 = 0;
          v10[1] = (__int64)(v10 + 3);
          v10[2] = 0x100000000LL;
          goto LABEL_7;
        }
LABEL_5:
        v12 = *((unsigned int *)v10 + 4);
        v13 = v10 + 1;
        v14 = v12 + 1;
        if ( v12 + 1 > (unsigned __int64)*((unsigned int *)v10 + 5) )
        {
          sub_C8D5F0((__int64)(v10 + 1), v10 + 3, v12 + 1, 8u, v14, v56);
          v12 = *((unsigned int *)v10 + 4);
        }
LABEL_7:
        ++v3;
        *(_QWORD *)(*(_QWORD *)v13 + 8 * v12) = v23;
        ++*((_DWORD *)v13 + 2);
        if ( v106 == v3 )
        {
          if ( (_DWORD)v116 )
          {
            v57 = v115;
            v58 = 4LL * v117;
            v107 = &v115[v58];
            if ( v115 != &v115[v58] )
            {
              while ( *v57 == -8192 || *v57 == -4096 )
              {
                v57 += 4;
                if ( v107 == v57 )
                  goto LABEL_31;
              }
              v108 = v57;
              if ( v107 != v57 )
              {
                while ( 1 )
                {
                  v118 = (_QWORD *)*v108;
                  v119 = v121;
                  v120 = 0x100000000LL;
                  v59 = *((_DWORD *)v108 + 4);
                  if ( !v59 || &v119 == v108 + 1 )
                  {
                    v60 = (__int64 *)v124;
                    v61 = 0;
                    v122 = (__int64 *)v124;
                    v123 = 0x300000000LL;
                    goto LABEL_74;
                  }
                  v76 = v121;
                  v77 = 8;
                  if ( v59 == 1
                    || (sub_C8D5F0((__int64)&v119, v121, v59, 8u, v14, v56),
                        v76 = v119,
                        (v77 = 8LL * *((unsigned int *)v108 + 4)) != 0) )
                  {
                    memcpy(v76, (const void *)v108[1], v77);
                    v76 = v119;
                  }
                  LODWORD(v120) = v59;
                  v109 = (__int64 *)&v76[8 * v59];
                  v111 = (__int64 *)v76;
                  v80 = a1;
                  v122 = (__int64 *)v124;
                  v123 = 0x300000000LL;
                  do
                  {
                    v81 = *v111;
                    v82 = *(_QWORD **)(*v111 + 24);
                    v83 = &v82[*(unsigned int *)(*v111 + 32)];
                    if ( v82 != v83 )
                    {
                      while ( 1 )
                      {
                        v84 = *v82;
                        v85 = v82;
                        if ( (*v82 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                        {
                          if ( *(_QWORD *)(*v82 & 0xFFFFFFFFFFFFFFF8LL) )
                            break;
                        }
                        if ( v83 == ++v82 )
                          goto LABEL_115;
                      }
LABEL_105:
                      if ( v83 == v85 )
                        goto LABEL_115;
                      v86 = *(_DWORD *)(v80 + 328);
                      v87 = v84 & 0xFFFFFFFFFFFFFFF8LL;
                      v88 = *(_QWORD *)(v80 + 312);
                      v89 = v87;
                      if ( !v86 )
                        goto LABEL_123;
                      v90 = v86 - 1;
                      v91 = v90 & (((unsigned int)v87 >> 9) ^ ((unsigned int)v87 >> 4));
                      v92 = (__int64 *)(v88 + 16LL * v91);
                      v93 = *v92;
                      if ( v89 == *v92 )
                        goto LABEL_108;
                      v99 = 1;
                      if ( v93 == -4096 )
                      {
LABEL_123:
                        v94 = 0;
                      }
                      else
                      {
                        while ( 1 )
                        {
                          v78 = (unsigned int)(v99 + 1);
                          v91 = v90 & (v99 + v91);
                          v92 = (__int64 *)(v88 + 16LL * v91);
                          v100 = *v92;
                          if ( v89 == *v92 )
                            break;
                          v99 = v78;
                          if ( v100 == -4096 )
                            goto LABEL_123;
                        }
LABEL_108:
                        v94 = (_QWORD *)v92[1];
                        if ( v94 )
                          v94 = (_QWORD *)*v94;
                      }
                      if ( v118 == v94 )
                      {
                        v96 = (unsigned int)v123;
                        v97 = (unsigned int)v123 + 1LL;
                        if ( v97 > HIDWORD(v123) )
                        {
                          sub_C8D5F0((__int64)&v122, v124, v97, 0x10u, v78, v79);
                          v96 = (unsigned int)v123;
                        }
                        v98 = &v122[2 * v96];
                        *v98 = v81;
                        v98[1] = v89;
                        LODWORD(v123) = v123 + 1;
                      }
                      else
                      {
                        sub_D23FE0((__int64)v118, v81, v89);
                      }
                      v95 = v85 + 1;
                      if ( v83 == v85 + 1 )
                        goto LABEL_115;
                      do
                      {
                        v84 = *v95;
                        v85 = v95;
                        if ( (*v95 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                        {
                          if ( *(_QWORD *)(*v95 & 0xFFFFFFFFFFFFFFF8LL) )
                            goto LABEL_105;
                        }
                        ++v95;
                      }
                      while ( v83 != v95 );
                    }
LABEL_115:
                    ++v111;
                  }
                  while ( v109 != v111 );
                  v60 = v122;
                  v61 = (unsigned int)v123;
                  a1 = v80;
LABEL_74:
                  v62 = (__int64)v118;
                  sub_D2C610(&v112, (__int64)v118, v60, v61);
                  if ( v112 != (__int64 **)&v113 )
                    _libc_free(v112, v62);
                  v63 = (__int64 *)v119;
                  v64 = &v119[8 * (unsigned int)v120];
                  if ( v64 != v119 )
                  {
                    while ( 1 )
                    {
                      v65 = *(_DWORD *)(a1 + 328);
                      v66 = *v63;
                      v62 = *(_QWORD *)(a1 + 312);
                      if ( !v65 )
                        goto LABEL_136;
                      v67 = v65 - 1;
                      v68 = (v65 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
                      v69 = (__int64 *)(v62 + 16LL * v68);
                      v70 = *v69;
                      if ( v66 != *v69 )
                        break;
LABEL_79:
                      v71 = (__int64 *)v69[1];
                      if ( !v71 )
                        goto LABEL_136;
                      v72 = *v71;
                      ++v63;
                      *(_DWORD *)(v72 + 16) = 0;
                      sub_D24EE0(v72 + 56);
                      *(_QWORD *)v72 = 0;
                      if ( v64 == (char *)v63 )
                        goto LABEL_81;
                    }
                    v74 = 1;
                    while ( v70 != -4096 )
                    {
                      v75 = v74 + 1;
                      v68 = v67 & (v74 + v68);
                      v69 = (__int64 *)(v62 + 16LL * v68);
                      v70 = *v69;
                      if ( v66 == *v69 )
                        goto LABEL_79;
                      v74 = v75;
                    }
LABEL_136:
                    MEMORY[0x10] = 0;
                    BUG();
                  }
LABEL_81:
                  if ( v122 != (__int64 *)v124 )
                    _libc_free(v122, v62);
                  if ( v119 != v121 )
                    _libc_free(v119, v62);
                  v73 = v108 + 4;
                  if ( v108 + 4 != v107 )
                  {
                    while ( *v73 == -8192 || *v73 == -4096 )
                    {
                      v73 += 4;
                      if ( v107 == v73 )
                        goto LABEL_31;
                    }
                    v108 = v73;
                    if ( v107 != v73 )
                      continue;
                  }
                  break;
                }
              }
            }
          }
LABEL_31:
          v32 = v105;
LABEL_33:
          v33 = *(_DWORD *)(a1 + 120);
          v34 = *v32;
          v35 = *(_QWORD *)(a1 + 104);
          if ( v33 )
          {
            v36 = v33 - 1;
            v37 = v36 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
            v38 = (__int64 *)(v35 + 16LL * v37);
            v39 = *v38;
            if ( v34 == *v38 )
            {
LABEL_35:
              v40 = v38[1];
              goto LABEL_36;
            }
            v55 = 1;
            while ( v39 != -4096 )
            {
              v102 = v55 + 1;
              v37 = v36 & (v55 + v37);
              v38 = (__int64 *)(v35 + 16LL * v37);
              v39 = *v38;
              if ( v34 == *v38 )
                goto LABEL_35;
              v55 = v102;
            }
          }
          v40 = 0;
LABEL_36:
          sub_D23E00(a1 + 128, v40);
          v41 = *(_DWORD *)(a1 + 328);
          v42 = *(_QWORD *)(a1 + 312);
          if ( v41 )
          {
            v43 = (v41 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
            v44 = (__int64 *)(v42 + 16LL * v43);
            v45 = *v44;
            if ( v40 == *v44 )
              goto LABEL_38;
            v54 = 1;
            while ( v45 != -4096 )
            {
              v101 = v54 + 1;
              v43 = (v41 - 1) & (v54 + v43);
              v44 = (__int64 *)(v42 + 16LL * v43);
              v45 = *v44;
              if ( v40 == *v44 )
                goto LABEL_38;
              v54 = v101;
            }
          }
          v44 = (__int64 *)(v42 + 16LL * v41);
LABEL_38:
          *v44 = -8192;
          a2 = *(unsigned int *)(a1 + 120);
          --*(_DWORD *)(a1 + 320);
          v46 = *(_QWORD *)(a1 + 104);
          ++*(_DWORD *)(a1 + 324);
          if ( (_DWORD)a2 )
          {
            v47 = (a2 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
            v48 = (__int64 *)(v46 + 16LL * v47);
            v49 = *v48;
            if ( v34 == *v48 )
            {
LABEL_40:
              *v48 = -8192;
              --*(_DWORD *)(a1 + 112);
              ++*(_DWORD *)(a1 + 116);
              if ( *(_BYTE *)(v40 + 104) )
              {
                v50 = *(unsigned int *)(v40 + 96);
                v51 = *(_QWORD *)(v40 + 80);
                *(_BYTE *)(v40 + 104) = 0;
                a2 = 16 * v50;
                sub_C7D6A0(v51, a2, 8);
                v52 = *(_QWORD *)(v40 + 24);
                if ( v52 != v40 + 40 )
                  _libc_free(v52, a2);
              }
              *(_QWORD *)v40 = 0;
              ++v32;
              *(_QWORD *)(v40 + 8) = 0;
              if ( v106 == v32 )
                break;
              goto LABEL_33;
            }
            v53 = 1;
            while ( v49 != -4096 )
            {
              v104 = v53 + 1;
              v47 = (a2 - 1) & (v53 + v47);
              v48 = (__int64 *)(v46 + 16LL * v47);
              v49 = *v48;
              if ( v34 == *v48 )
                goto LABEL_40;
              v53 = v104;
            }
          }
          v48 = (__int64 *)(v46 + 16LL * (unsigned int)a2);
          goto LABEL_40;
        }
        v5 = (__int64)v115;
      }
    }
    v28 = v117;
    if ( v117 )
    {
      v29 = v115;
      v30 = &v115[4 * v117];
      do
      {
        if ( *v29 != -4096 && *v29 != -8192 )
        {
          v31 = (_QWORD *)v29[1];
          if ( v31 != v29 + 3 )
            _libc_free(v31, a2);
        }
        v29 += 4;
      }
      while ( v30 != v29 );
      v28 = v117;
    }
    sub_C7D6A0((__int64)v115, 32 * v28, 8);
  }
}
