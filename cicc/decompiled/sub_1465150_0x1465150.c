// Function: sub_1465150
// Address: 0x1465150
//
void __fastcall sub_1465150(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // eax
  __int64 v5; // rax
  int v6; // r8d
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 v9; // rsi
  __int64 v10; // rcx
  unsigned __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  int v20; // r8d
  __int64 v21; // rcx
  unsigned int v22; // edx
  __int64 v23; // rsi
  __int64 v24; // rcx
  unsigned __int64 v25; // r14
  __int64 v26; // r15
  __int64 v27; // r12
  __int64 v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // r13
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rsi
  unsigned int v35; // edx
  int v36; // edi
  __int64 *v37; // rbx
  __int64 v38; // rcx
  __int64 *v39; // r15
  unsigned __int64 v40; // r14
  __int64 v41; // rsi
  __int64 i; // rbx
  char v43; // dl
  __int64 v44; // r13
  _QWORD *v45; // rax
  _QWORD *v46; // rdi
  _QWORD *v47; // rsi
  int v48; // r14d
  __int64 v49; // r15
  char v50; // r10
  unsigned int v51; // ebx
  int v52; // edx
  unsigned int v53; // edi
  __int64 v54; // rax
  __int64 v55; // r14
  __int64 v56; // r9
  int v57; // eax
  int v58; // edx
  __int64 v59; // rsi
  unsigned int v60; // ebx
  __int64 *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // r13
  __int64 v64; // rax
  int v65; // eax
  int v66; // esi
  __int64 v67; // rcx
  unsigned int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // rdi
  __int64 v71; // r14
  __int64 v72; // rax
  __int64 v73; // r13
  __int64 v74; // rbx
  _QWORD *v75; // rcx
  __int64 v76; // rax
  unsigned int v77; // ebx
  __int64 v78; // rcx
  _QWORD *v79; // rdx
  _QWORD *v80; // r15
  _QWORD *v81; // rax
  _QWORD *v82; // r14
  _QWORD *v83; // rbx
  unsigned __int64 v84; // rdi
  __int64 v85; // r11
  int v86; // eax
  int v87; // eax
  int v88; // ecx
  int v89; // eax
  int v90; // r8d
  int v91; // r8d
  __int64 v92; // rax
  __int64 v93; // r8
  __int64 v94; // [rsp+8h] [rbp-318h]
  __int64 v95; // [rsp+8h] [rbp-318h]
  __int64 v96; // [rsp+10h] [rbp-310h]
  __int64 v97; // [rsp+10h] [rbp-310h]
  __int64 v98; // [rsp+10h] [rbp-310h]
  __int64 *v99; // [rsp+38h] [rbp-2E8h]
  __int64 *v100; // [rsp+38h] [rbp-2E8h]
  char v101; // [rsp+38h] [rbp-2E8h]
  char v102; // [rsp+38h] [rbp-2E8h]
  __int64 v103; // [rsp+38h] [rbp-2E8h]
  void *v104; // [rsp+40h] [rbp-2E0h] BYREF
  char v105[16]; // [rsp+48h] [rbp-2D8h] BYREF
  __int64 v106; // [rsp+58h] [rbp-2C8h]
  void *v107; // [rsp+70h] [rbp-2B0h] BYREF
  char v108[16]; // [rsp+78h] [rbp-2A8h] BYREF
  __int64 v109; // [rsp+88h] [rbp-298h]
  _QWORD *v110; // [rsp+A0h] [rbp-280h] BYREF
  unsigned int v111; // [rsp+A8h] [rbp-278h]
  unsigned int v112; // [rsp+ACh] [rbp-274h]
  _QWORD v113[16]; // [rsp+B0h] [rbp-270h] BYREF
  __int64 v114; // [rsp+130h] [rbp-1F0h] BYREF
  _BYTE *v115; // [rsp+138h] [rbp-1E8h]
  _BYTE *v116; // [rsp+140h] [rbp-1E0h]
  __int64 v117; // [rsp+148h] [rbp-1D8h]
  int v118; // [rsp+150h] [rbp-1D0h]
  _BYTE v119[136]; // [rsp+158h] [rbp-1C8h] BYREF
  _BYTE *v120; // [rsp+1E0h] [rbp-140h] BYREF
  __int64 v121; // [rsp+1E8h] [rbp-138h]
  _BYTE v122[304]; // [rsp+1F0h] [rbp-130h] BYREF

  v2 = a2;
  v110 = v113;
  v120 = v122;
  v112 = 16;
  v113[0] = a2;
  v114 = 0;
  v117 = 16;
  v118 = 0;
  v121 = 0x2000000000LL;
  v115 = v119;
  v116 = v119;
  v4 = 1;
  while ( 2 )
  {
    v111 = v4 - 1;
    v5 = *(unsigned int *)(a1 + 552);
    if ( (_DWORD)v5 )
    {
      v6 = 1;
      v7 = *(_QWORD *)(a1 + 536);
      v8 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v99 = (__int64 *)(v7 + ((unsigned __int64)v8 << 6));
      v9 = *v99;
      if ( *v99 == v2 )
      {
LABEL_4:
        if ( v99 != (__int64 *)(v7 + (v5 << 6)) )
        {
          sub_14575E0(v99 + 1);
          v10 = v99[1];
          v11 = v10 + 24LL * *((unsigned int *)v99 + 4);
          if ( v10 != v11 )
          {
            v96 = a1;
            v12 = v99[1];
            v94 = v2;
            do
            {
              v13 = *(_QWORD *)(v11 - 8);
              v11 -= 24LL;
              if ( v13 )
              {
                *(_QWORD *)v13 = &unk_49EC708;
                v14 = *(unsigned int *)(v13 + 208);
                if ( (_DWORD)v14 )
                {
                  v15 = *(_QWORD **)(v13 + 192);
                  v16 = &v15[7 * v14];
                  do
                  {
                    if ( *v15 != -8 && *v15 != -16 )
                    {
                      v17 = v15[1];
                      if ( (_QWORD *)v17 != v15 + 3 )
                        _libc_free(v17);
                    }
                    v15 += 7;
                  }
                  while ( v16 != v15 );
                }
                j___libc_free_0(*(_QWORD *)(v13 + 192));
                v18 = *(_QWORD *)(v13 + 40);
                if ( v18 != v13 + 56 )
                  _libc_free(v18);
                j_j___libc_free_0(v13, 216);
              }
            }
            while ( v12 != v11 );
            a1 = v96;
            v2 = v94;
            v11 = v99[1];
          }
          if ( (__int64 *)v11 != v99 + 3 )
            _libc_free(v11);
          *v99 = -16;
          --*(_DWORD *)(a1 + 544);
          ++*(_DWORD *)(a1 + 548);
        }
      }
      else
      {
        while ( v9 != -8 )
        {
          v8 = (v5 - 1) & (v6 + v8);
          v99 = (__int64 *)(v7 + ((unsigned __int64)v8 << 6));
          v9 = *v99;
          if ( *v99 == v2 )
            goto LABEL_4;
          ++v6;
        }
      }
    }
    v19 = *(unsigned int *)(a1 + 584);
    if ( (_DWORD)v19 )
    {
      v20 = 1;
      v21 = *(_QWORD *)(a1 + 568);
      v22 = (v19 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v100 = (__int64 *)(v21 + ((unsigned __int64)v22 << 6));
      v23 = *v100;
      if ( v2 == *v100 )
      {
LABEL_25:
        if ( v100 != (__int64 *)(v21 + (v19 << 6)) )
        {
          sub_14575E0(v100 + 1);
          v24 = v100[1];
          v25 = v24 + 24LL * *((unsigned int *)v100 + 4);
          if ( v24 != v25 )
          {
            v97 = a1;
            v26 = v100[1];
            v95 = v2;
            do
            {
              v27 = *(_QWORD *)(v25 - 8);
              v25 -= 24LL;
              if ( v27 )
              {
                *(_QWORD *)v27 = &unk_49EC708;
                v28 = *(unsigned int *)(v27 + 208);
                if ( (_DWORD)v28 )
                {
                  v29 = *(_QWORD **)(v27 + 192);
                  v30 = &v29[7 * v28];
                  do
                  {
                    if ( *v29 != -8 && *v29 != -16 )
                    {
                      v31 = v29[1];
                      if ( (_QWORD *)v31 != v29 + 3 )
                        _libc_free(v31);
                    }
                    v29 += 7;
                  }
                  while ( v30 != v29 );
                }
                j___libc_free_0(*(_QWORD *)(v27 + 192));
                v32 = *(_QWORD *)(v27 + 40);
                if ( v32 != v27 + 56 )
                  _libc_free(v32);
                j_j___libc_free_0(v27, 216);
              }
            }
            while ( v26 != v25 );
            a1 = v97;
            v2 = v95;
            v25 = v100[1];
          }
          if ( (__int64 *)v25 != v100 + 3 )
            _libc_free(v25);
          *v100 = -16;
          --*(_DWORD *)(a1 + 576);
          ++*(_DWORD *)(a1 + 580);
        }
      }
      else
      {
        while ( v23 != -8 )
        {
          v22 = (v19 - 1) & (v20 + v22);
          v100 = (__int64 *)(v21 + ((unsigned __int64)v22 << 6));
          v23 = *v100;
          if ( *v100 == v2 )
            goto LABEL_25;
          ++v20;
        }
      }
    }
    if ( *(_DWORD *)(a1 + 1016) )
    {
      v78 = *(unsigned int *)(a1 + 1024);
      v79 = *(_QWORD **)(a1 + 1008);
      v80 = &v79[8 * v78];
      if ( v79 != v80 )
      {
        v81 = *(_QWORD **)(a1 + 1008);
        while ( 1 )
        {
          while ( 1 )
          {
            v82 = v81;
            if ( *v81 != -8 )
              break;
            if ( v81[1] != -8 )
              goto LABEL_105;
            v81 += 8;
            if ( v80 == v81 )
              goto LABEL_45;
          }
          if ( *v81 != -16 || v81[1] != -16 )
            break;
          v81 += 8;
          if ( v80 == v81 )
            goto LABEL_45;
        }
LABEL_105:
        if ( v80 != v81 )
        {
          do
          {
            v83 = v82 + 8;
            if ( v82[1] == v2 )
            {
              while ( v80 != v83 )
              {
                if ( *v83 == -8 )
                {
                  if ( v83[1] != -8 )
                    break;
                  v83 += 8;
                }
                else
                {
                  if ( *v83 != -16 || v83[1] != -16 )
                    break;
                  v83 += 8;
                }
              }
              v84 = v82[3];
              if ( (_QWORD *)v84 != v82 + 5 )
                _libc_free(v84);
              *v82 = -16;
              v82[1] = -16;
              v82 = v83;
              v79 = *(_QWORD **)(a1 + 1008);
              --*(_DWORD *)(a1 + 1016);
              v78 = *(unsigned int *)(a1 + 1024);
              ++*(_DWORD *)(a1 + 1020);
            }
            else
            {
              while ( v80 != v83 )
              {
                if ( *v83 == -8 )
                {
                  if ( v83[1] != -8 )
                    goto LABEL_110;
                  v83 += 8;
                }
                else
                {
                  if ( *v83 != -16 || v83[1] != -16 )
                  {
LABEL_110:
                    v82 = v83;
                    goto LABEL_111;
                  }
                  v83 += 8;
                }
              }
              v82 = v80;
            }
LABEL_111:
            ;
          }
          while ( v82 != &v79[8 * v78] );
        }
      }
    }
LABEL_45:
    v33 = *(unsigned int *)(a1 + 992);
    if ( (_DWORD)v33 )
    {
      v34 = *(_QWORD *)(a1 + 976);
      v35 = (v33 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v36 = 1;
      v37 = (__int64 *)(v34 + 56LL * v35);
      v38 = *v37;
      if ( v2 == *v37 )
      {
LABEL_47:
        if ( v37 != (__int64 *)(v34 + 56 * v33) )
        {
          v39 = (__int64 *)v37[1];
          v40 = (unsigned __int64)&v39[*((unsigned int *)v37 + 4)];
          if ( v39 != (__int64 *)v40 )
          {
            do
            {
              v41 = *v39++;
              sub_1459590(a1, v41);
            }
            while ( (__int64 *)v40 != v39 );
            v40 = v37[1];
          }
          if ( (__int64 *)v40 != v37 + 3 )
            _libc_free(v40);
          *v37 = -16;
          --*(_DWORD *)(a1 + 984);
          ++*(_DWORD *)(a1 + 988);
        }
      }
      else
      {
        while ( v38 != -8 )
        {
          v35 = (v33 - 1) & (v36 + v35);
          v37 = (__int64 *)(v34 + 56LL * v35);
          v38 = *v37;
          if ( *v37 == v2 )
            goto LABEL_47;
          ++v36;
        }
      }
    }
    sub_1453C80(v2, (__int64)&v120);
    if ( !(_DWORD)v121 )
      goto LABEL_88;
    v98 = v2;
    LODWORD(i) = v121;
    do
    {
      v44 = *(_QWORD *)&v120[8 * (unsigned int)i - 8];
      LODWORD(v121) = i - 1;
      v45 = v115;
      if ( v116 != v115 )
        goto LABEL_56;
      v46 = &v115[8 * HIDWORD(v117)];
      if ( v115 != (_BYTE *)v46 )
      {
        v47 = 0;
        while ( v44 != *v45 )
        {
          if ( *v45 == -2 )
            v47 = v45;
          if ( v46 == ++v45 )
          {
            if ( !v47 )
              goto LABEL_99;
            *v47 = v44;
            --v118;
            ++v114;
            goto LABEL_68;
          }
        }
LABEL_57:
        LODWORD(i) = v121;
        continue;
      }
LABEL_99:
      if ( HIDWORD(v117) < (unsigned int)v117 )
      {
        ++HIDWORD(v117);
        *v46 = v44;
        ++v114;
      }
      else
      {
LABEL_56:
        sub_16CCBA0(&v114, v44);
        if ( !v43 )
          goto LABEL_57;
      }
LABEL_68:
      v48 = *(_DWORD *)(a1 + 168);
      if ( v48 )
      {
        v49 = *(_QWORD *)(a1 + 152);
        sub_1457D90(&v104, -8, 0);
        sub_1457D90(&v107, -16, 0);
        v50 = 1;
        v51 = ((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4);
        v52 = v48 - 1;
        v53 = (v48 - 1) & v51;
        v54 = 48LL * v53;
        v55 = v49 + v54;
        v56 = *(_QWORD *)(v49 + v54 + 24);
        if ( v44 != v56 )
        {
          v85 = v49 + v54;
          v86 = 1;
          v55 = 0;
          while ( v56 != v106 )
          {
            if ( v56 != v109 || v55 )
              v85 = v55;
            v91 = v86 + 1;
            v92 = v52 & (v53 + v86);
            v53 = v92;
            v55 = v49 + 48 * v92;
            v56 = *(_QWORD *)(v55 + 24);
            if ( v44 == v56 )
            {
              v50 = 1;
              goto LABEL_70;
            }
            v86 = v91;
            v93 = v85;
            v85 = v55;
            v55 = v93;
          }
          v50 = 0;
          if ( !v55 )
            v55 = v85;
        }
LABEL_70:
        v107 = &unk_49EE2B0;
        if ( v109 != -8 && v109 != 0 && v109 != -16 )
        {
          v101 = v50;
          sub_1649B30(v108);
          v50 = v101;
        }
        v104 = &unk_49EE2B0;
        if ( v106 != -8 && v106 != 0 && v106 != -16 )
        {
          v102 = v50;
          sub_1649B30(v105);
          v50 = v102;
        }
        if ( v50 )
        {
          if ( v55 != *(_QWORD *)(a1 + 152) + 48LL * *(unsigned int *)(a1 + 168) )
          {
            sub_1464220(a1, *(_QWORD *)(v55 + 24));
            sub_1459590(a1, *(_QWORD *)(v55 + 40));
            if ( *(_BYTE *)(v44 + 16) == 77 )
            {
              v57 = *(_DWORD *)(a1 + 616);
              if ( v57 )
              {
                v58 = v57 - 1;
                v59 = *(_QWORD *)(a1 + 600);
                v60 = (v57 - 1) & v51;
                v61 = (__int64 *)(v59 + 16LL * v60);
                v62 = *v61;
                if ( v44 == *v61 )
                {
LABEL_81:
                  *v61 = -16;
                  --*(_DWORD *)(a1 + 608);
                  ++*(_DWORD *)(a1 + 612);
                }
                else
                {
                  v87 = 1;
                  while ( v62 != -8 )
                  {
                    v88 = v87 + 1;
                    v60 = v58 & (v87 + v60);
                    v61 = (__int64 *)(v59 + 16LL * v60);
                    v62 = *v61;
                    if ( v44 == *v61 )
                      goto LABEL_81;
                    v87 = v88;
                  }
                }
              }
            }
          }
        }
      }
      v63 = *(_QWORD *)(v44 + 8);
      for ( i = (unsigned int)v121; v63; v63 = *(_QWORD *)(v63 + 8) )
      {
        v64 = sub_1648700(v63);
        if ( HIDWORD(v121) <= (unsigned int)i )
        {
          v103 = v64;
          sub_16CD150(&v120, v122, 0, 8);
          i = (unsigned int)v121;
          v64 = v103;
        }
        *(_QWORD *)&v120[8 * i] = v64;
        i = (unsigned int)(v121 + 1);
        LODWORD(v121) = v121 + 1;
      }
    }
    while ( (_DWORD)i );
    v2 = v98;
LABEL_88:
    v65 = *(_DWORD *)(a1 + 712);
    if ( v65 )
    {
      v66 = v65 - 1;
      v67 = *(_QWORD *)(a1 + 696);
      v68 = (v65 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v69 = (__int64 *)(v67 + 16LL * v68);
      v70 = *v69;
      if ( v2 == *v69 )
      {
LABEL_90:
        *v69 = -16;
        --*(_DWORD *)(a1 + 704);
        ++*(_DWORD *)(a1 + 708);
      }
      else
      {
        v89 = 1;
        while ( v70 != -8 )
        {
          v90 = v89 + 1;
          v68 = v66 & (v89 + v68);
          v69 = (__int64 *)(v67 + 16LL * v68);
          v70 = *v69;
          if ( v2 == *v69 )
            goto LABEL_90;
          v89 = v90;
        }
      }
    }
    v71 = *(_QWORD *)(v2 + 8);
    v72 = v111;
    v73 = *(_QWORD *)(v2 + 16) - v71;
    v74 = v73 >> 3;
    if ( v73 >> 3 > v112 - (unsigned __int64)v111 )
    {
      sub_16CD150(&v110, v113, v74 + v111, 8);
      v72 = v111;
    }
    v75 = &v110[v72];
    if ( v73 > 0 )
    {
      v76 = 0;
      do
      {
        v75[v76] = *(_QWORD *)(v71 + 8 * v76);
        ++v76;
      }
      while ( v74 - v76 > 0 );
      LODWORD(v72) = v111;
    }
    v77 = v72 + v74;
    v111 = v77;
    v4 = v77;
    if ( v77 )
    {
      v2 = v110[v77 - 1];
      continue;
    }
    break;
  }
  if ( v116 != v115 )
    _libc_free((unsigned __int64)v116);
  if ( v120 != v122 )
    _libc_free((unsigned __int64)v120);
  if ( v110 != v113 )
    _libc_free((unsigned __int64)v110);
}
