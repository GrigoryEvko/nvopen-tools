// Function: sub_B1F4E0
// Address: 0xb1f4e0
//
__int64 __fastcall sub_B1F4E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  _QWORD *v4; // r14
  __int64 v6; // r13
  __int64 *v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r15
  int v11; // r9d
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // r13
  __int64 v15; // rax
  int v16; // edi
  __int64 *v17; // r13
  __int64 v18; // rax
  __int64 *v19; // r12
  __int64 v20; // r15
  __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // r10
  unsigned __int64 v26; // rdx
  __int64 *v27; // rax
  unsigned int v28; // r8d
  __int64 v29; // rax
  __int64 *v30; // r12
  __int64 v31; // r13
  unsigned int v32; // r14d
  __int64 v33; // rdx
  unsigned int v34; // eax
  __int64 v35; // rbx
  __int64 *v36; // r15
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // r12
  __int64 v47; // r15
  __int64 v48; // r8
  __int64 v49; // rax
  _QWORD *v50; // rax
  _BYTE *v51; // rdx
  __int64 v52; // r9
  __int64 *v53; // rax
  __int64 v54; // r10
  __int64 v55; // r11
  __int64 *v56; // rdx
  __int64 v57; // r9
  __int64 v58; // r8
  _BYTE *v59; // r14
  _BYTE *v60; // rdi
  __int64 *v61; // rax
  __int64 v62; // rax
  _BYTE *v63; // r15
  _BYTE *v64; // r13
  _BYTE *v65; // rdi
  __int64 v66; // r13
  unsigned int v67; // ebx
  _BYTE *v68; // rsi
  __int64 v69; // rax
  unsigned __int64 v70; // r13
  unsigned int i; // eax
  __int64 v72; // rdx
  __int64 v73; // r15
  int v74; // r10d
  __int64 v75; // rax
  int v76; // r10d
  __int64 v77; // rdx
  int v78; // edx
  __int64 *v79; // r10
  __int64 *v80; // r15
  _QWORD *v81; // r8
  unsigned int v82; // eax
  unsigned int v83; // r14d
  __int64 v84; // rcx
  unsigned int v85; // edx
  __int64 v86; // rdx
  __int64 v87; // rdx
  unsigned __int64 v88; // r9
  __int64 *v89; // rdx
  __int64 v90; // rbx
  __int64 v91; // rbx
  __int64 v92; // rax
  __int64 *v93; // rdx
  __int64 *v94; // rbx
  __int64 v95; // rdx
  unsigned int v96; // eax
  __int64 v97; // r13
  __int64 v98; // rax
  __int64 v99; // rdx
  unsigned int v100; // eax
  __int64 v101; // r15
  __int64 v102; // r14
  _QWORD *v103; // rdi
  _QWORD *v104; // rax
  int v105; // r10d
  size_t v106; // rdx
  __int64 v107; // rax
  _BYTE *v108; // rbx
  __int64 result; // rax
  _BYTE *v110; // r12
  _BYTE *v111; // rdi
  unsigned __int64 v112; // [rsp+8h] [rbp-15C8h]
  _QWORD *v114; // [rsp+10h] [rbp-15C0h]
  __int64 *v115; // [rsp+20h] [rbp-15B0h]
  unsigned int v117; // [rsp+48h] [rbp-1588h]
  int v118; // [rsp+48h] [rbp-1588h]
  unsigned int v119; // [rsp+58h] [rbp-1578h]
  __int64 v120; // [rsp+58h] [rbp-1578h]
  __int64 *v121; // [rsp+58h] [rbp-1578h]
  int v122; // [rsp+60h] [rbp-1570h]
  __int64 *v123; // [rsp+60h] [rbp-1570h]
  unsigned int v124; // [rsp+60h] [rbp-1570h]
  __int64 v125; // [rsp+60h] [rbp-1570h]
  unsigned __int64 v126; // [rsp+60h] [rbp-1570h]
  unsigned int v127; // [rsp+68h] [rbp-1568h]
  __int64 v128; // [rsp+68h] [rbp-1568h]
  int v129; // [rsp+68h] [rbp-1568h]
  __int64 *v130; // [rsp+68h] [rbp-1568h]
  unsigned __int64 v131; // [rsp+68h] [rbp-1568h]
  __int64 v132; // [rsp+68h] [rbp-1568h]
  __int64 v133; // [rsp+78h] [rbp-1558h] BYREF
  __int64 *v134; // [rsp+80h] [rbp-1550h] BYREF
  int v135; // [rsp+88h] [rbp-1548h]
  _BYTE v136[64]; // [rsp+90h] [rbp-1540h] BYREF
  __int64 *v137; // [rsp+D0h] [rbp-1500h] BYREF
  __int64 v138; // [rsp+D8h] [rbp-14F8h]
  _BYTE v139[128]; // [rsp+E0h] [rbp-14F0h] BYREF
  _BYTE *v140; // [rsp+160h] [rbp-1470h] BYREF
  unsigned int v141; // [rsp+168h] [rbp-1468h]
  unsigned int v142; // [rsp+16Ch] [rbp-1464h]
  _BYTE v143[1024]; // [rsp+170h] [rbp-1460h] BYREF
  _QWORD *v144; // [rsp+570h] [rbp-1060h] BYREF
  __int64 v145; // [rsp+578h] [rbp-1058h]
  _QWORD v146[64]; // [rsp+580h] [rbp-1050h] BYREF
  _BYTE *v147; // [rsp+780h] [rbp-E50h]
  __int64 v148; // [rsp+788h] [rbp-E48h]
  _BYTE v149[3584]; // [rsp+790h] [rbp-E40h] BYREF
  __int64 v150; // [rsp+1590h] [rbp-40h]

  v4 = &v144;
  v6 = *(_QWORD *)a3;
  v137 = (__int64 *)v139;
  v138 = 0x1000000000LL;
  v119 = *(_DWORD *)(a3 + 16);
  v144 = v146;
  v145 = 0x4000000001LL;
  v147 = v149;
  v150 = a2;
  v148 = 0x4000000000LL;
  v146[0] = 0;
  v134 = (__int64 *)v6;
  v135 = 0;
  sub_B1C510(&v140, &v134, 1);
  v7 = (__int64 *)v6;
  v127 = 0;
  *(_DWORD *)(sub_B1E0B0((__int64)&v144, v6) + 4) = 0;
  v8 = v141;
  if ( v141 )
  {
    do
    {
      while ( 1 )
      {
        v9 = (__int64)&v140[16 * v8 - 16];
        v10 = *(_QWORD *)v9;
        v11 = *(_DWORD *)(v9 + 8);
        v141 = v8 - 1;
        v7 = (__int64 *)v10;
        v122 = v11;
        v12 = sub_B1E0B0((__int64)&v144, v10);
        v13 = v122;
        v14 = v12;
        v15 = *(unsigned int *)(v12 + 32);
        if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v14 + 36) )
        {
          v7 = (__int64 *)(v14 + 40);
          sub_C8D5F0(v14 + 24, v14 + 40, v15 + 1, 4);
          v15 = *(unsigned int *)(v14 + 32);
          v13 = v122;
        }
        *(_DWORD *)(*(_QWORD *)(v14 + 24) + 4 * v15) = v13;
        v16 = *(_DWORD *)v14;
        ++*(_DWORD *)(v14 + 32);
        if ( !v16 )
        {
          *(_DWORD *)(v14 + 4) = v13;
          *(_DWORD *)(v14 + 12) = ++v127;
          *(_DWORD *)(v14 + 8) = v127;
          *(_DWORD *)v14 = v127;
          sub_B1A4E0((__int64)&v144, v10);
          v7 = (__int64 *)v10;
          sub_B1D150(&v134, v10, v150);
          v17 = &v134[v135];
          if ( v134 != v17 )
          {
            v18 = a1;
            v19 = v134;
            v20 = v18;
            do
            {
              while ( 1 )
              {
                v23 = *v19;
                v133 = v23;
                if ( v23 )
                {
                  v21 = (unsigned int)(*(_DWORD *)(v23 + 44) + 1);
                  v22 = *(_DWORD *)(v23 + 44) + 1;
                }
                else
                {
                  v22 = 0;
                  v21 = 0;
                }
                if ( v22 >= *(_DWORD *)(v20 + 32) )
LABEL_121:
                  BUG();
                if ( v119 < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v20 + 24) + 8 * v21) + 16LL) )
                  break;
                v7 = &v137[(unsigned int)v138];
                if ( v7 == sub_B18540(v137, (__int64)v7, &v133) )
                {
                  v7 = (__int64 *)v23;
                  sub_B1A4E0((__int64)&v137, v23);
                }
                if ( v17 == ++v19 )
                  goto LABEL_20;
              }
              v24 = v141;
              v25 = v3 & 0xFFFFFFFF00000000LL | v127;
              v26 = v141 + 1LL;
              v3 = v25;
              if ( v26 > v142 )
              {
                v7 = (__int64 *)v143;
                v126 = v25;
                sub_C8D5F0(&v140, v143, v26, 16);
                v24 = v141;
                v25 = v126;
              }
              ++v19;
              v27 = (__int64 *)&v140[16 * v24];
              *v27 = v23;
              v27[1] = v25;
              ++v141;
            }
            while ( v17 != v19 );
LABEL_20:
            v17 = v134;
            a1 = v20;
          }
          if ( v17 != (__int64 *)v136 )
            break;
        }
        v8 = v141;
        if ( !v141 )
          goto LABEL_23;
      }
      _libc_free(v17, v7);
      v8 = v141;
    }
    while ( v141 );
LABEL_23:
    v4 = &v144;
  }
  if ( v140 != v143 )
    _libc_free(v140, v7);
  if ( v137 == &v137[(unsigned int)v138] )
  {
    v42 = a3;
  }
  else
  {
    v28 = *(_DWORD *)(a1 + 32);
    v29 = a1;
    v123 = &v137[(unsigned int)v138];
    v30 = v137;
    v31 = v29;
    v120 = a3;
    v32 = v28;
    do
    {
      v41 = *v30;
      if ( *v30 )
      {
        v33 = (unsigned int)(*(_DWORD *)(v41 + 44) + 1);
        v34 = *(_DWORD *)(v41 + 44) + 1;
      }
      else
      {
        v33 = 0;
        v34 = 0;
      }
      if ( v34 >= v32 )
        BUG();
      v35 = *(_QWORD *)(v31 + 24);
      v36 = *(__int64 **)(v35 + 8 * v33);
      v37 = sub_B192F0(v31, *v36, *(_QWORD *)a3);
      if ( v37 )
      {
        v38 = (unsigned int)(*(_DWORD *)(v37 + 44) + 1);
        v39 = *(_DWORD *)(v37 + 44) + 1;
      }
      else
      {
        v38 = 0;
        v39 = 0;
      }
      if ( v39 >= v32 )
        goto LABEL_121;
      v40 = *(_QWORD *)(v35 + 8 * v38);
      if ( (__int64 *)v40 != v36 )
      {
        if ( *(_DWORD *)(v120 + 16) <= *(_DWORD *)(v40 + 16) )
          v40 = v120;
        v120 = v40;
      }
      ++v30;
    }
    while ( v123 != v30 );
    v42 = v120;
    v4 = &v144;
    a1 = v31;
  }
  if ( !*(_QWORD *)(v42 + 8) )
  {
    v43 = a2;
    sub_B1EF50(a1, a2);
LABEL_118:
    result = sub_B1AC50((__int64)&v144, v43);
    goto LABEL_114;
  }
  v43 = v127;
  if ( v127 )
  {
    v44 = v127;
    v128 = v42;
    v45 = a1;
    v46 = 8 * v44;
    v47 = 8 * (v44 - (unsigned int)(v44 - 1));
    while ( 1 )
    {
      v48 = 0;
      v49 = v144[(unsigned __int64)v46 / 8];
      if ( v49 )
        v48 = 8LL * (unsigned int)(*(_DWORD *)(v49 + 44) + 1);
      v50 = (_QWORD *)(v48 + *(_QWORD *)(v45 + 24));
      v51 = (_BYTE *)*v50;
      *(_BYTE *)(v45 + 112) = 0;
      v140 = v51;
      v52 = *((_QWORD *)v51 + 1);
      if ( v52 )
      {
        v53 = sub_B186E0(
                *(_QWORD **)(v52 + 24),
                *(_QWORD *)(v52 + 24) + 8LL * *(unsigned int *)(v52 + 32),
                (__int64 *)&v140);
        v56 = (__int64 *)(v55 + v54 - 8);
        v43 = *v53;
        *v53 = *v56;
        *v56 = v43;
        --*(_DWORD *)(v57 + 32);
        v50 = (_QWORD *)(v58 + *(_QWORD *)(v45 + 24));
      }
      v59 = (_BYTE *)*v50;
      *v50 = 0;
      if ( v59 )
      {
        v60 = (_BYTE *)*((_QWORD *)v59 + 3);
        if ( v60 != v59 + 40 )
          _libc_free(v60, v43);
        v43 = 80;
        j_j___libc_free_0(v59, 80);
      }
      if ( v47 == v46 )
        break;
      v46 -= 8;
    }
    a1 = v45;
    v4 = &v144;
    v42 = v128;
  }
  if ( a3 == v42 )
    goto LABEL_118;
  v124 = *(_DWORD *)(v42 + 16);
  v61 = *(__int64 **)(v42 + 8);
  LODWORD(v145) = 0;
  v121 = v61;
  v62 = 0;
  if ( !HIDWORD(v145) )
  {
    v43 = (__int64)v146;
    sub_C8D5F0(&v144, v146, 1, 8);
    v62 = (unsigned int)v145;
  }
  v144[v62] = 0;
  LODWORD(v145) = v145 + 1;
  v63 = v147;
  v64 = &v147[56 * (unsigned int)v148];
  if ( v147 != v64 )
  {
    do
    {
      v64 -= 56;
      v65 = (_BYTE *)*((_QWORD *)v64 + 3);
      if ( v65 != v64 + 40 )
        _libc_free(v65, v43);
    }
    while ( v63 != v64 );
  }
  LODWORD(v148) = 0;
  v66 = *(_QWORD *)v42;
  v67 = 0;
  v135 = 0;
  v134 = (__int64 *)v66;
  sub_B1C510(&v140, &v134, 1);
  v68 = (_BYTE *)v66;
  v69 = sub_B1E0B0((__int64)&v144, v66);
  v70 = v112;
  *(_DWORD *)(v69 + 4) = 0;
  for ( i = v141; v141; i = v141 )
  {
    while ( 1 )
    {
      v72 = (__int64)&v140[16 * i - 16];
      v73 = *(_QWORD *)v72;
      v74 = *(_DWORD *)(v72 + 8);
      v141 = i - 1;
      v68 = (_BYTE *)v73;
      v129 = v74;
      v75 = sub_B1E0B0((__int64)v4, v73);
      v76 = v129;
      v77 = *(unsigned int *)(v75 + 32);
      if ( v77 + 1 > (unsigned __int64)*(unsigned int *)(v75 + 36) )
      {
        v68 = (_BYTE *)(v75 + 40);
        v118 = v129;
        v132 = v75;
        sub_C8D5F0(v75 + 24, v75 + 40, v77 + 1, 4);
        v75 = v132;
        v76 = v118;
        v77 = *(unsigned int *)(v132 + 32);
      }
      *(_DWORD *)(*(_QWORD *)(v75 + 24) + 4 * v77) = v76;
      v78 = *(_DWORD *)v75;
      ++*(_DWORD *)(v75 + 32);
      if ( !v78 )
      {
        ++v67;
        *(_DWORD *)(v75 + 4) = v76;
        *(_DWORD *)(v75 + 12) = v67;
        *(_DWORD *)(v75 + 8) = v67;
        *(_DWORD *)v75 = v67;
        sub_B1A4E0((__int64)v4, v73);
        v68 = (_BYTE *)v73;
        sub_B1D150(&v134, v73, v150);
        v79 = &v134[v135];
        if ( v134 != v79 )
        {
          v80 = v134;
          v81 = v4;
          v82 = v124;
          v83 = v67;
          do
          {
            v90 = *v80;
            if ( *v80 )
            {
              v84 = (unsigned int)(*(_DWORD *)(v90 + 44) + 1);
              v85 = *(_DWORD *)(v90 + 44) + 1;
            }
            else
            {
              v85 = 0;
              v84 = 0;
            }
            if ( v85 < *(_DWORD *)(a1 + 32) )
            {
              v86 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v84);
              if ( v86 )
              {
                if ( v82 < *(_DWORD *)(v86 + 16) )
                {
                  v87 = v141;
                  v68 = (_BYTE *)0xFFFFFFFF00000000LL;
                  v88 = v70 & 0xFFFFFFFF00000000LL | v83;
                  v70 = v88;
                  if ( (unsigned __int64)v141 + 1 > v142 )
                  {
                    v68 = v143;
                    v114 = v81;
                    v115 = v79;
                    v117 = v82;
                    v131 = v88;
                    sub_C8D5F0(&v140, v143, v141 + 1LL, 16);
                    v87 = v141;
                    v81 = v114;
                    v79 = v115;
                    v82 = v117;
                    v88 = v131;
                  }
                  v89 = (__int64 *)&v140[16 * v87];
                  *v89 = v90;
                  v89[1] = v88;
                  ++v141;
                }
              }
            }
            ++v80;
          }
          while ( v79 != v80 );
          v79 = v134;
          v67 = v83;
          v4 = v81;
        }
        if ( v79 != (__int64 *)v136 )
          break;
      }
      i = v141;
      if ( !v141 )
        goto LABEL_84;
    }
    _libc_free(v79, v68);
  }
LABEL_84:
  if ( v140 != v143 )
    _libc_free(v140, v68);
  sub_B1E260((__int64)v4);
  v91 = *v121;
  v43 = 1;
  *(_QWORD *)(sub_B1E0B0((__int64)v4, v144[1]) + 16) = v91;
  v92 = sub_B1B2D0(v4, 1);
  v130 = v93;
  v94 = (__int64 *)v92;
  if ( (__int64 *)v92 != v93 )
  {
    v125 = (__int64)v4;
    do
    {
      v43 = *v94;
      if ( *v94 )
      {
        v95 = (unsigned int)(*(_DWORD *)(v43 + 44) + 1);
        v96 = *(_DWORD *)(v43 + 44) + 1;
      }
      else
      {
        v95 = 0;
        v96 = 0;
      }
      v97 = 0;
      if ( v96 < *(_DWORD *)(a1 + 32) )
        v97 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v95);
      v98 = *(_QWORD *)(sub_B1E0B0(v125, v43) + 16);
      if ( v98 )
      {
        v99 = (unsigned int)(*(_DWORD *)(v98 + 44) + 1);
        v100 = *(_DWORD *)(v98 + 44) + 1;
      }
      else
      {
        v99 = 0;
        v100 = 0;
      }
      v101 = 0;
      if ( v100 < *(_DWORD *)(a1 + 32) )
        v101 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v99);
      v102 = *(_QWORD *)(v97 + 8);
      if ( v101 != v102 )
      {
        v140 = (_BYTE *)v97;
        v103 = *(_QWORD **)(v102 + 24);
        v43 = (__int64)&v103[*(unsigned int *)(v102 + 32)];
        v104 = sub_B186E0(v103, v43, (__int64 *)&v140);
        if ( v104 + 1 != (_QWORD *)v43 )
        {
          v106 = v43 - (_QWORD)(v104 + 1);
          v43 = (__int64)(v104 + 1);
          memmove(v104, v104 + 1, v106);
          v105 = *(_DWORD *)(v102 + 32);
        }
        *(_DWORD *)(v102 + 32) = v105 - 1;
        *(_QWORD *)(v97 + 8) = v101;
        v107 = *(unsigned int *)(v101 + 32);
        if ( v107 + 1 > (unsigned __int64)*(unsigned int *)(v101 + 36) )
        {
          v43 = v101 + 40;
          sub_C8D5F0(v101 + 24, v101 + 40, v107 + 1, 8);
          v107 = *(unsigned int *)(v101 + 32);
        }
        *(_QWORD *)(*(_QWORD *)(v101 + 24) + 8 * v107) = v97;
        ++*(_DWORD *)(v101 + 32);
        sub_B19190(v97, (_QWORD *)v43);
      }
      ++v94;
    }
    while ( v130 != v94 );
  }
  v108 = v147;
  result = 7LL * (unsigned int)v148;
  v110 = &v147[56 * (unsigned int)v148];
  if ( v147 != v110 )
  {
    do
    {
      v110 -= 56;
      v111 = (_BYTE *)*((_QWORD *)v110 + 3);
      result = (__int64)(v110 + 40);
      if ( v111 != v110 + 40 )
        result = _libc_free(v111, v43);
    }
    while ( v108 != v110 );
    v110 = v147;
  }
  if ( v110 != v149 )
    result = _libc_free(v110, v43);
  if ( v144 != v146 )
    result = _libc_free(v144, v43);
LABEL_114:
  if ( v137 != (__int64 *)v139 )
    return _libc_free(v137, v43);
  return result;
}
