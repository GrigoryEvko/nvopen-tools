// Function: sub_34D4DC0
// Address: 0x34d4dc0
//
unsigned __int64 __fastcall sub_34D4DC0(
        __int64 a1,
        int a2,
        _QWORD **a3,
        unsigned int a4,
        unsigned int *a5,
        signed __int64 a6,
        unsigned __int8 a7,
        unsigned int a8,
        int a9,
        char a10,
        char a11)
{
  unsigned __int64 v11; // r14
  unsigned int v13; // r12d
  unsigned int v14; // ebx
  __int64 v15; // rcx
  int v16; // edx
  _QWORD *v17; // r14
  __int64 v18; // rcx
  unsigned __int16 v19; // r15
  __int64 v20; // rdx
  __int64 v21; // r13
  __int16 v22; // r9
  __int64 v23; // r13
  int v24; // r14d
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // r15d
  unsigned __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // r9
  __int64 v31; // r8
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned int *i; // r9
  unsigned int v35; // ecx
  unsigned int v36; // edx
  __int64 v37; // r11
  unsigned int v38; // esi
  char v39; // al
  int v40; // r14d
  unsigned int v41; // r15d
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rax
  __int64 *v44; // rsi
  unsigned int v45; // eax
  signed __int64 v46; // rdx
  __int64 v47; // rcx
  unsigned __int64 v48; // rax
  int v49; // r14d
  unsigned int v50; // r15d
  unsigned __int64 v51; // rbx
  unsigned __int64 v52; // rax
  __int64 *v53; // rsi
  unsigned int v54; // eax
  bool v55; // of
  __int64 v56; // rdx
  int v57; // edx
  unsigned int v58; // r10d
  unsigned int v59; // r15d
  unsigned int v60; // r14d
  _QWORD *v61; // r9
  unsigned int *v62; // r15
  unsigned int v63; // esi
  unsigned int v64; // edi
  unsigned int v65; // eax
  unsigned int v66; // r14d
  __int64 *v67; // r15
  __int64 *v68; // rbx
  __int64 v69; // rdi
  int v70; // r14d
  unsigned int v71; // r15d
  unsigned __int64 v72; // r12
  unsigned __int64 v73; // rax
  __int64 *v74; // rsi
  unsigned int v75; // eax
  signed __int64 v76; // rdx
  signed __int64 v77; // rax
  unsigned __int64 v78; // rdx
  bool v79; // cc
  unsigned __int64 v80; // rax
  __int64 *v81; // rax
  __int64 *v82; // r15
  void **v83; // r8
  signed __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  int v87; // r14d
  unsigned int v88; // r15d
  unsigned __int64 v89; // rax
  __int64 *v90; // rsi
  unsigned int v91; // eax
  size_t v92; // rdx
  __int64 v93; // rdx
  __int64 v94; // [rsp+8h] [rbp-108h]
  _QWORD *v95; // [rsp+10h] [rbp-100h]
  unsigned int v96; // [rsp+10h] [rbp-100h]
  __int16 v97; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v98; // [rsp+20h] [rbp-F0h]
  __int64 v99; // [rsp+20h] [rbp-F0h]
  __int64 v100; // [rsp+20h] [rbp-F0h]
  __int64 v101; // [rsp+20h] [rbp-F0h]
  unsigned int v102; // [rsp+20h] [rbp-F0h]
  __int64 v103; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v104; // [rsp+30h] [rbp-E0h]
  unsigned int v105; // [rsp+38h] [rbp-D8h]
  unsigned int v107; // [rsp+48h] [rbp-C8h]
  unsigned int v109; // [rsp+50h] [rbp-C0h]
  unsigned int v111; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v112; // [rsp+58h] [rbp-B8h]
  unsigned int v113; // [rsp+58h] [rbp-B8h]
  int v114; // [rsp+60h] [rbp-B0h]
  unsigned int v115; // [rsp+64h] [rbp-ACh]
  unsigned __int64 v117; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v118; // [rsp+78h] [rbp-98h]
  unsigned __int64 v119; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v120; // [rsp+88h] [rbp-88h]
  void *v121; // [rsp+90h] [rbp-80h] BYREF
  __int64 v122; // [rsp+98h] [rbp-78h]
  _QWORD v123[6]; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v124; // [rsp+D0h] [rbp-40h]

  if ( *((_BYTE *)a3 + 8) == 18 )
    return 0;
  v13 = a4;
  v115 = *((_DWORD *)a3 + 8);
  v14 = v115 / a4;
  v103 = sub_BCDA70(a3[3], v115 / a4);
  if ( a10 || a11 )
  {
    v104 = sub_34D46A0(a1, a2, a3, a7, 1, 0, a9, 0);
    v114 = v57;
  }
  else
  {
    v15 = a7;
    BYTE1(v15) = 1;
    v104 = sub_34D2F80(a1, a2, (__int64)a3, v15, a8, a9);
    v114 = v16;
  }
  v17 = *a3;
  v18 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), (__int64 *)a3, 0);
  v19 = v18;
  v21 = v20;
  while ( 1 )
  {
    LOWORD(v18) = v19;
    sub_2FE6CC0((__int64)&v121, *(_QWORD *)(a1 + 24), (__int64)v17, v18, v21);
    v22 = v122;
    if ( (_BYTE)v121 == 10 )
      break;
    if ( !(_BYTE)v121 )
    {
      v23 = a1;
      goto LABEL_141;
    }
    if ( v19 == (_WORD)v122 )
    {
      if ( (_WORD)v122 )
      {
        v23 = a1;
        v24 = (unsigned __int16)v122;
        goto LABEL_13;
      }
      if ( v21 == v123[0] )
      {
        v23 = a1;
        v24 = 0;
        goto LABEL_13;
      }
    }
    v18 = v122;
    v21 = v123[0];
    v19 = v122;
  }
  v23 = a1;
  if ( !v19 )
  {
    v24 = 8;
    v22 = 8;
    goto LABEL_13;
  }
LABEL_141:
  v24 = v19;
  v22 = v19;
LABEL_13:
  v97 = v22;
  v25 = sub_9208B0(*(_QWORD *)(v23 + 8), (__int64)a3);
  v122 = v26;
  v121 = (void *)((unsigned __int64)(v25 + 7) >> 3);
  v27 = sub_CA1930(&v121);
  if ( (unsigned __int16)v97 <= 1u || (unsigned __int16)(v97 - 504) <= 7u )
    BUG();
  v28 = *(_QWORD *)&byte_444C4A0[16 * v24 - 16] + 7LL;
  LOBYTE(v122) = byte_444C4A0[16 * v24 - 8];
  v121 = (void *)(v28 >> 3);
  v29 = sub_CA1930(&v121);
  v31 = (__int64)&a5[a6];
  if ( !v114 && v27 > v29 )
  {
    v58 = (v27 - (v27 != 0)) / v29 + (v27 != 0);
    v59 = (v58 + 63) >> 6;
    v60 = (v115 - (v115 != 0)) / v58 + (v115 != 0);
    v121 = v123;
    v122 = 0x600000000LL;
    if ( v59 > 6 )
    {
      v96 = v58;
      sub_C8D5F0((__int64)&v121, v123, v59, 8u, v31, v30);
      memset(v121, 0, 8LL * v59);
      LODWORD(v122) = v59;
      v61 = v121;
      v58 = v96;
      v31 = (__int64)&a5[a6];
    }
    else
    {
      if ( v59 )
      {
        v92 = 8LL * v59;
        if ( v92 )
        {
          v102 = v58;
          memset(v123, 0, v92);
          v31 = (__int64)&a5[a6];
          v58 = v102;
        }
      }
      LODWORD(v122) = v59;
      v61 = v123;
    }
    v124 = v58;
    if ( a5 != (unsigned int *)v31 )
    {
      v62 = a5;
      do
      {
        v63 = *v62;
        if ( v13 <= v115 )
        {
          v64 = 0;
          while ( 1 )
          {
            ++v64;
            v65 = v63 / v60;
            v63 += v13;
            v61[v65 >> 6] |= 1LL << v65;
            if ( v14 <= v64 )
              break;
            v61 = v121;
          }
          v61 = v121;
        }
        ++v62;
      }
      while ( (unsigned int *)v31 != v62 );
    }
    v98 = v58;
    if ( &v61[(unsigned int)v122] == v61 )
    {
      v104 = 0;
    }
    else
    {
      v95 = v61;
      v66 = 0;
      v67 = &v61[(unsigned int)v122];
      v94 = v31;
      v105 = v14;
      v68 = v61;
      do
      {
        v69 = *v68++;
        v66 += sub_39FAC40(v69);
      }
      while ( v67 != v68 );
      v61 = v95;
      v31 = v94;
      v14 = v105;
      v104 = (v104 * v66 != 0) + (v104 * v66 - (v104 * v66 != 0)) / v98;
    }
    if ( v61 != v123 )
    {
      v99 = v31;
      _libc_free((unsigned __int64)v61);
      v31 = v99;
    }
  }
  v118 = v14;
  if ( v14 > 0x40 )
  {
    v100 = v31;
    sub_C43690((__int64)&v117, -1, 1);
    v31 = v100;
  }
  else
  {
    v32 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
    if ( v13 > v115 )
      v32 = 0;
    v117 = v32;
  }
  v120 = v115;
  if ( v115 > 0x40 )
  {
    v101 = v31;
    sub_C43690((__int64)&v119, -1, 1);
    LODWORD(v122) = v115;
    sub_C43690((__int64)&v121, 0, 0);
    v31 = v101;
  }
  else
  {
    LODWORD(v122) = v115;
    v121 = 0;
    v33 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v115;
    if ( !v115 )
      v33 = 0;
    v119 = v33;
  }
  for ( i = a5; i != (unsigned int *)v31; ++i )
  {
    v35 = *i;
    v36 = 0;
    if ( v13 <= v115 )
    {
      do
      {
        while ( 1 )
        {
          v37 = 1LL << v35;
          if ( (unsigned int)v122 > 0x40 )
            break;
          ++v36;
          v35 += v13;
          v121 = (void *)(v37 | (unsigned __int64)v121);
          if ( v14 <= v36 )
            goto LABEL_31;
        }
        v38 = v35;
        ++v36;
        v35 += v13;
        *((_QWORD *)v121 + (v38 >> 6)) |= v37;
      }
      while ( v14 > v36 );
    }
LABEL_31:
    ;
  }
  v39 = *(_BYTE *)(v103 + 8);
  if ( a2 != 32 )
  {
    if ( v39 == 18 )
    {
      v48 = v104;
      goto LABEL_48;
    }
    v40 = *(_DWORD *)(v103 + 32);
    if ( v40 <= 0 )
    {
      v48 = v104;
      goto LABEL_48;
    }
    v111 = v13;
    v41 = 0;
    v42 = 0;
    do
    {
      v43 = v117;
      if ( v118 > 0x40 )
        v43 = *(_QWORD *)(v117 + 8LL * (v41 >> 6));
      if ( (v43 & (1LL << v41)) != 0 )
      {
        v44 = (__int64 *)v103;
        if ( (unsigned int)*(unsigned __int8 *)(v103 + 8) - 17 <= 1 )
          v44 = **(__int64 ***)(v103 + 16);
        v45 = sub_34D06B0(v23, v44);
        if ( __OFADD__(v45, v42) )
        {
          v42 = 0x8000000000000000LL;
          if ( v45 )
            v42 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v42 += v45;
        }
      }
      ++v41;
    }
    while ( v40 != v41 );
    v46 = v42;
    v13 = v111;
    v47 = v46 * a6;
    if ( is_mul_ok(v46, a6) )
    {
      v48 = v47 + v104;
      if ( __OFADD__(v47, v104) )
      {
        v48 = 0x8000000000000000LL;
        if ( v47 > 0 )
          v48 = 0x7FFFFFFFFFFFFFFFLL;
      }
      goto LABEL_48;
    }
    if ( v46 <= 0 )
    {
      if ( v46 < 0 && a6 < 0 )
      {
        v48 = 0x7FFFFFFFFFFFFFFFLL;
        v93 = v104 + 0x7FFFFFFFFFFFFFFFLL;
        if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v104) )
          goto LABEL_168;
        goto LABEL_48;
      }
    }
    else if ( a6 > 0 )
    {
      v93 = 0x7FFFFFFFFFFFFFFFLL;
      v48 = v104 + 0x7FFFFFFFFFFFFFFFLL;
      if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v104) )
      {
LABEL_168:
        v112 = v93;
LABEL_49:
        if ( *((_BYTE *)a3 + 8) != 18 )
        {
          v49 = *((_DWORD *)a3 + 8);
          if ( v49 > 0 )
          {
            v109 = v13;
            v50 = 0;
            v107 = v14;
            v51 = 0;
            do
            {
              v52 = (unsigned __int64)v121;
              if ( (unsigned int)v122 > 0x40 )
                v52 = *((_QWORD *)v121 + (v50 >> 6));
              if ( (v52 & (1LL << v50)) != 0 )
              {
                v53 = (__int64 *)a3;
                if ( (unsigned int)*((unsigned __int8 *)a3 + 8) - 17 <= 1 )
                  v53 = (__int64 *)*a3[2];
                v54 = sub_34D06B0(v23, v53);
                v55 = __OFADD__(v54, v51);
                v51 += v54;
                if ( v55 )
                {
                  v51 = 0x8000000000000000LL;
                  if ( v54 )
                    v51 = 0x7FFFFFFFFFFFFFFFLL;
                }
              }
              ++v50;
            }
            while ( v49 != v50 );
LABEL_59:
            v56 = v51;
            v13 = v109;
            v14 = v107;
            v11 = v56 + v112;
            if ( __OFADD__(v56, v112) )
            {
              v11 = 0x7FFFFFFFFFFFFFFFLL;
              if ( v56 <= 0 )
                v11 = 0x8000000000000000LL;
            }
            goto LABEL_62;
          }
LABEL_137:
          v11 = v112;
          goto LABEL_62;
        }
        goto LABEL_132;
      }
LABEL_48:
      v112 = v48;
      goto LABEL_49;
    }
    v48 = 0x8000000000000000LL;
    v93 = v104 + 0x8000000000000000LL;
    if ( !__OFADD__(0x8000000000000000LL, v104) )
      goto LABEL_168;
    goto LABEL_48;
  }
  if ( v39 == 18 )
  {
    v78 = v104;
    goto LABEL_119;
  }
  v70 = *(_DWORD *)(v103 + 32);
  if ( v70 <= 0 )
  {
    v78 = v104;
    goto LABEL_119;
  }
  v113 = v13;
  v71 = 0;
  v72 = 0;
  do
  {
    v73 = v117;
    if ( v118 > 0x40 )
      v73 = *(_QWORD *)(v117 + 8LL * (v71 >> 6));
    if ( (v73 & (1LL << v71)) != 0 )
    {
      v74 = (__int64 *)v103;
      if ( (unsigned int)*(unsigned __int8 *)(v103 + 8) - 17 <= 1 )
        v74 = **(__int64 ***)(v103 + 16);
      v75 = sub_34D06B0(v23, v74);
      if ( __OFADD__(v75, v72) )
      {
        v72 = 0x8000000000000000LL;
        if ( v75 )
          v72 = 0x7FFFFFFFFFFFFFFFLL;
      }
      else
      {
        v72 += v75;
      }
    }
    ++v71;
  }
  while ( v70 != v71 );
  v76 = v72;
  v13 = v113;
  v77 = v76 * a6;
  if ( !is_mul_ok(v76, a6) )
  {
    if ( a6 <= 0 )
    {
      if ( v76 < 0 && a6 < 0 )
      {
LABEL_156:
        v80 = 0x7FFFFFFFFFFFFFFFLL;
        v78 = v104 + 0x7FFFFFFFFFFFFFFFLL;
        if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v104) )
          goto LABEL_119;
LABEL_107:
        v112 = v80;
        goto LABEL_120;
      }
    }
    else if ( v76 > 0 )
    {
      goto LABEL_156;
    }
    v80 = 0x8000000000000000LL;
    v78 = v104 + 0x8000000000000000LL;
    if ( !__OFADD__(0x8000000000000000LL, v104) )
      goto LABEL_119;
    goto LABEL_107;
  }
  v78 = v77 + v104;
  if ( __OFADD__(v77, v104) )
  {
    v79 = v77 <= 0;
    v80 = 0x8000000000000000LL;
    if ( !v79 )
      v80 = 0x7FFFFFFFFFFFFFFFLL;
    goto LABEL_107;
  }
LABEL_119:
  v112 = v78;
LABEL_120:
  if ( *((_BYTE *)a3 + 8) != 18 )
  {
    v87 = *((_DWORD *)a3 + 8);
    if ( v87 > 0 )
    {
      v109 = v13;
      v88 = 0;
      v107 = v14;
      v51 = 0;
      do
      {
        v89 = (unsigned __int64)v121;
        if ( (unsigned int)v122 > 0x40 )
          v89 = *((_QWORD *)v121 + (v88 >> 6));
        if ( (v89 & (1LL << v88)) != 0 )
        {
          v90 = (__int64 *)a3;
          if ( (unsigned int)*((unsigned __int8 *)a3 + 8) - 17 <= 1 )
            v90 = (__int64 *)*a3[2];
          v91 = sub_34D06B0(v23, v90);
          v55 = __OFADD__(v91, v51);
          v51 += v91;
          if ( v55 )
          {
            v51 = 0x8000000000000000LL;
            if ( v91 )
              v51 = 0x7FFFFFFFFFFFFFFFLL;
          }
        }
        ++v88;
      }
      while ( v87 != v88 );
      goto LABEL_59;
    }
    goto LABEL_137;
  }
LABEL_132:
  v11 = v112;
LABEL_62:
  if ( a10 )
  {
    v81 = (__int64 *)sub_BCB2B0(*a3);
    v82 = v81;
    v83 = (void **)&v119;
    if ( a11 )
      v83 = &v121;
    v84 = sub_34D1730(v23, v81, v13, v14, (__int64)v83);
    v55 = __OFADD__(v84, v11);
    v11 += v84;
    if ( v55 )
    {
      v11 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v84 <= 0 )
        v11 = 0x8000000000000000LL;
    }
    if ( a11 )
    {
      v85 = sub_BCDA70(v82, v115);
      v86 = sub_34D2250(v23, 0x1Cu, v85, a9, 0, 0, 0, 0, 0);
      v55 = __OFADD__(v86, v11);
      v11 += v86;
      if ( v55 )
      {
        v11 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v86 <= 0 )
          v11 = 0x8000000000000000LL;
      }
    }
  }
  if ( (unsigned int)v122 > 0x40 && v121 )
    j_j___libc_free_0_0((unsigned __int64)v121);
  if ( v120 > 0x40 && v119 )
    j_j___libc_free_0_0(v119);
  if ( v118 > 0x40 && v117 )
    j_j___libc_free_0_0(v117);
  return v11;
}
