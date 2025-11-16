// Function: sub_3388570
// Address: 0x3388570
//
__int64 __fastcall sub_3388570(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        _BYTE *a5,
        unsigned __int8 **a6,
        __int64 a7,
        int a8)
{
  _BYTE *v11; // r11
  __int64 v13; // rdi
  bool v14; // al
  int v15; // ecx
  int v16; // edx
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 *v22; // r9
  unsigned __int64 *v23; // r8
  int v24; // edi
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r14
  void (__fastcall *v28)(__int64 **, __int64, __int64); // r13
  __int64 v29; // rax
  __int64 v30; // r9
  _BYTE *v31; // r12
  unsigned __int64 v32; // rbx
  __int64 v33; // r13
  __int64 v34; // rax
  int v35; // r8d
  __int64 *v36; // rdi
  __int64 *v37; // r10
  unsigned int v38; // esi
  __int64 *v39; // rcx
  _BYTE *v40; // r11
  _BYTE *v41; // rdx
  unsigned int v42; // r8d
  __int64 *v43; // rcx
  unsigned __int8 **v44; // r12
  signed __int64 v45; // rbx
  unsigned __int8 *v46; // r15
  __int64 v47; // r8
  unsigned __int8 *v48; // r11
  unsigned __int8 *v49; // r8
  __int64 v50; // r8
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  unsigned __int8 **v53; // rdi
  int v54; // ecx
  unsigned __int8 **v55; // r9
  unsigned __int64 v56; // rcx
  int v57; // edx
  __int64 v58; // r15
  bool v59; // of
  int v60; // ecx
  int v61; // r8d
  unsigned int v62; // edx
  __int64 *v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rdx
  int v66; // r10d
  char *v67; // rax
  int v68; // r10d
  unsigned int v69; // esi
  __int64 *v70; // rdx
  __int64 v71; // r8
  _QWORD *v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rsi
  unsigned __int64 v75; // rax
  char v76; // r8
  __int64 *v77; // rsi
  __int64 *v78; // rdx
  __int64 *v79; // rcx
  __int64 v80; // r9
  unsigned __int64 v81; // r8
  __int64 v82; // r9
  __int64 *v83; // rdx
  bool v84; // al
  __int64 v85; // rax
  int v86; // eax
  int v87; // edx
  int v88; // eax
  int v89; // esi
  __int64 v90; // rax
  int v91; // edi
  int v92; // [rsp+10h] [rbp-300h]
  int v93; // [rsp+20h] [rbp-2F0h]
  __int64 v94; // [rsp+20h] [rbp-2F0h]
  __int64 v95; // [rsp+28h] [rbp-2E8h]
  __int64 v96; // [rsp+28h] [rbp-2E8h]
  __int64 v97; // [rsp+30h] [rbp-2E0h]
  __int64 v98; // [rsp+38h] [rbp-2D8h]
  unsigned int v99; // [rsp+38h] [rbp-2D8h]
  unsigned __int8 *v100; // [rsp+38h] [rbp-2D8h]
  bool v101; // [rsp+47h] [rbp-2C9h]
  unsigned __int64 v103; // [rsp+48h] [rbp-2C8h]
  int v104; // [rsp+48h] [rbp-2C8h]
  __int64 v105; // [rsp+48h] [rbp-2C8h]
  unsigned __int8 v106; // [rsp+50h] [rbp-2C0h]
  __int64 v107; // [rsp+50h] [rbp-2C0h]
  char *v109; // [rsp+58h] [rbp-2B8h]
  __int64 *v110; // [rsp+68h] [rbp-2A8h] BYREF
  unsigned __int8 **v111; // [rsp+70h] [rbp-2A0h] BYREF
  __int64 v112; // [rsp+78h] [rbp-298h]
  _BYTE v113[32]; // [rsp+80h] [rbp-290h] BYREF
  __int64 v114; // [rsp+A0h] [rbp-270h] BYREF
  __int64 v115; // [rsp+A8h] [rbp-268h]
  __int64 *v116; // [rsp+B0h] [rbp-260h] BYREF
  unsigned int v117; // [rsp+B8h] [rbp-258h]
  unsigned __int64 v118[2]; // [rsp+130h] [rbp-1E0h] BYREF
  _BYTE v119[128]; // [rsp+140h] [rbp-1D0h] BYREF
  __int64 v120; // [rsp+1C0h] [rbp-150h] BYREF
  __int64 v121; // [rsp+1C8h] [rbp-148h]
  __int64 *v122; // [rsp+1D0h] [rbp-140h] BYREF
  unsigned int v123; // [rsp+1D8h] [rbp-138h]
  _BYTE *v124; // [rsp+250h] [rbp-C0h] BYREF
  __int64 v125; // [rsp+258h] [rbp-B8h]
  _BYTE v126[176]; // [rsp+260h] [rbp-B0h] BYREF

  v101 = (int)a7 < 0 || (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) != 3;
  if ( v101 )
    return 0;
  v97 = (int)a7;
  v11 = a5;
  if ( __PAIR64__(a8, HIDWORD(a7)) )
  {
    v13 = *(_QWORD *)(a2 + 32);
    if ( v13 )
    {
      v107 = *(_QWORD *)(a2 + 32);
      v98 = *(_QWORD *)(a3 - 32);
      v14 = sub_FF06D0(v13, *(_QWORD *)(a3 + 40), *(_QWORD *)(a3 - 64));
      v15 = a4;
      v16 = 28;
      v11 = a5;
      if ( !v14 )
      {
        v84 = sub_FF06D0(v107, *(_QWORD *)(a3 + 40), v98);
        v11 = a5;
        if ( !v84 )
          goto LABEL_10;
        v15 = a4;
        v16 = 29;
      }
      if ( v15 == v16 )
      {
        v97 = SHIDWORD(a7) + (__int64)(int)a7;
      }
      else
      {
        if ( a8 < 0 )
          return 0;
        v97 = (int)a7 - (__int64)a8;
      }
    }
  }
LABEL_10:
  if ( v97 <= 0 )
    return 0;
  v114 = 0;
  v115 = 1;
  v17 = (unsigned __int64 *)&v116;
  do
  {
    *v17 = -4096;
    v17 += 2;
  }
  while ( v17 != v118 );
  v120 = 0;
  v121 = 1;
  v118[0] = (unsigned __int64)v119;
  v118[1] = 0x800000000LL;
  v18 = (unsigned __int64 *)&v122;
  do
  {
    *v18 = -4096;
    v18 += 2;
  }
  while ( v18 != (unsigned __int64 *)&v124 );
  v124 = v126;
  v125 = 0x800000000LL;
  sub_33883E0((__int64)&v114, v11, 0, 0, (__int64)v118, (__int64)&v116);
  v106 = sub_33883E0((__int64)&v120, a6, (__int64)&v114, 0, v19, v20);
  if ( !v106 )
    goto LABEL_53;
  v22 = (__int64 *)&v116;
  v23 = v118;
  if ( *(_BYTE *)a6 <= 0x1Cu )
    goto LABEL_21;
  v111 = a6;
  if ( (v115 & 1) != 0 )
  {
    v24 = 7;
  }
  else
  {
    v22 = v116;
    v23 = (unsigned __int64 *)&v116[2 * v117];
    if ( !v117 )
      goto LABEL_130;
    v24 = v117 - 1;
  }
  v25 = v24 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
  v26 = &v22[2 * v25];
  v21 = *v26;
  if ( a6 != (unsigned __int8 **)*v26 )
  {
    v88 = 1;
    while ( v21 != -4096 )
    {
      v89 = v88 + 1;
      v90 = v24 & (v25 + v88);
      v25 = v90;
      v26 = &v22[2 * v90];
      v21 = *v26;
      if ( a6 == (unsigned __int8 **)*v26 )
        goto LABEL_20;
      v88 = v89;
    }
    goto LABEL_130;
  }
LABEL_20:
  if ( v23 == (unsigned __int64 *)v26 )
  {
LABEL_130:
    LOBYTE(v110) = 0;
    sub_3388010((__int64)&v120, (__int64 *)&v111, &v110, v21, (__int64)v23, (__int64)v22);
  }
LABEL_21:
  v27 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL) + 8LL);
  v28 = *(void (__fastcall **)(__int64 **, __int64, __int64))(*(_QWORD *)v27 + 104LL);
  v29 = sub_B43CB0(a3);
  v28(&v110, v27, v29);
  v93 = 6;
  v31 = *(_BYTE **)(a3 - 96);
  v99 = v125;
  v103 = (unsigned __int64)v124;
  v95 = (unsigned int)v125;
  do
  {
    v109 = (char *)(v103 + 16 * v95);
    if ( v109 == (char *)v103 )
      goto LABEL_51;
    v32 = v103;
    while ( 1 )
    {
      v33 = *(_QWORD *)v32;
      v34 = *(_QWORD *)(*(_QWORD *)v32 + 16LL);
      if ( v34 )
        break;
LABEL_35:
      v32 += 16LL;
      if ( v109 == (char *)v32 )
        goto LABEL_36;
    }
    while ( 1 )
    {
      v41 = *(_BYTE **)(v34 + 24);
      if ( *v41 <= 0x1Cu || v31 == v41 )
        goto LABEL_29;
      v30 = v121 & 1;
      if ( (v121 & 1) != 0 )
      {
        v35 = 7;
        v36 = (__int64 *)&v124;
        v37 = (__int64 *)&v122;
      }
      else
      {
        v37 = v122;
        v42 = v123;
        v43 = v122;
        v36 = &v122[2 * v123];
        if ( !v123 )
          goto LABEL_67;
        v35 = v123 - 1;
      }
      v38 = v35 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v39 = &v37[2 * v38];
      v40 = (_BYTE *)*v39;
      if ( v41 != (_BYTE *)*v39 )
        break;
LABEL_28:
      if ( v39 == v36 )
        goto LABEL_65;
LABEL_29:
      v34 = *(_QWORD *)(v34 + 8);
      if ( !v34 )
        goto LABEL_35;
    }
    v60 = 1;
    while ( v40 != (_BYTE *)-4096LL )
    {
      v38 = v35 & (v60 + v38);
      v92 = v60 + 1;
      v39 = &v37[2 * v38];
      v40 = (_BYTE *)*v39;
      if ( v41 == (_BYTE *)*v39 )
        goto LABEL_28;
      v60 = v92;
    }
LABEL_65:
    if ( !(_BYTE)v30 )
    {
      v43 = v122;
      v42 = v123;
LABEL_67:
      if ( v42 )
      {
        v61 = v42 - 1;
        goto LABEL_69;
      }
LABEL_117:
      v85 = 2LL * v42;
      goto LABEL_118;
    }
    v61 = 7;
    v43 = (__int64 *)&v122;
LABEL_69:
    v62 = v61 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
    v63 = &v43[2 * v62];
    v64 = *v63;
    if ( v33 == *v63 )
      goto LABEL_70;
    v86 = 1;
    while ( v64 != -4096 )
    {
      v91 = v86 + 1;
      v62 = v61 & (v86 + v62);
      v63 = &v43[2 * v62];
      v64 = *v63;
      if ( v33 == *v63 )
        goto LABEL_70;
      v86 = v91;
    }
    if ( !(_BYTE)v30 )
    {
      v42 = v123;
      goto LABEL_117;
    }
    v85 = 16;
LABEL_118:
    v63 = &v43[v85];
LABEL_70:
    if ( (_BYTE)v30 )
    {
      v65 = 16;
      v66 = 8;
    }
    else
    {
      v66 = v123;
      v65 = 2LL * v123;
    }
    if ( v63 != &v43[v65] )
    {
      v67 = (char *)(v103 + 16LL * *((unsigned int *)v63 + 2));
      if ( v109 != v67 )
      {
        if ( (_BYTE)v30 || v123 )
        {
          v68 = v66 - 1;
          v69 = v68 & (((unsigned int)*(_QWORD *)v67 >> 9) ^ ((unsigned int)*(_QWORD *)v67 >> 4));
          v70 = &v43[2 * v69];
          v71 = *v70;
          if ( *v70 == *(_QWORD *)v67 )
          {
LABEL_77:
            *v70 = -8192;
            ++HIDWORD(v121);
            v103 = (unsigned __int64)v124;
            LODWORD(v121) = (2 * ((unsigned int)v121 >> 1) - 2) | v121 & 1;
            v99 = v125;
            v109 = &v124[16 * (unsigned int)v125];
          }
          else
          {
            v87 = 1;
            while ( v71 != -4096 )
            {
              v30 = (unsigned int)(v87 + 1);
              v69 = v68 & (v87 + v69);
              v70 = &v43[2 * v69];
              v71 = *v70;
              if ( *(_QWORD *)v67 == *v70 )
                goto LABEL_77;
              v87 = v30;
            }
          }
        }
        v72 = v67 + 16;
        v73 = (v109 - (v67 + 16)) >> 4;
        if ( v109 - (v67 + 16) > 0 )
        {
          do
          {
            v74 = *v72;
            v72 += 2;
            *(v72 - 4) = v74;
            *((_BYTE *)v72 - 24) = *((_BYTE *)v72 - 8);
            --v73;
          }
          while ( v73 );
          v103 = (unsigned __int64)v124;
          v99 = v125;
        }
        LODWORD(v125) = --v99;
        v95 = v99;
        v109 = (char *)(v103 + 16LL * v99);
        if ( v67 != v109 )
        {
          v75 = (__int64)&v67[-v103] >> 4;
          v76 = v121 & 1;
          if ( !((unsigned int)v121 >> 1) )
          {
            if ( v76 )
            {
              v82 = 16;
              v83 = (__int64 *)&v122;
            }
            else
            {
              v83 = v122;
              v82 = 2LL * v123;
            }
            v78 = &v83[v82];
            v77 = v78;
            goto LABEL_88;
          }
          if ( v76 )
          {
            v77 = (__int64 *)&v124;
            v78 = (__int64 *)&v122;
            do
            {
LABEL_85:
              if ( *v78 != -4096 && *v78 != -8192 )
                break;
              v78 += 2;
            }
            while ( v77 != v78 );
LABEL_88:
            if ( !v76 )
            {
              v79 = v122;
              v80 = v123;
              goto LABEL_90;
            }
            v30 = 128;
            v79 = (__int64 *)&v122;
          }
          else
          {
            v80 = v123;
            v79 = v122;
            v78 = v122;
            v77 = &v122[2 * v123];
            if ( v122 != v77 )
              goto LABEL_85;
LABEL_90:
            v30 = 16 * v80;
          }
          if ( (__int64 *)((char *)v79 + v30) != v78 )
          {
            do
            {
              v81 = *((unsigned int *)v78 + 2);
              if ( v75 < v81 )
                *((_DWORD *)v78 + 2) = v81 - 1;
              do
                v78 += 2;
              while ( v77 != v78 && (*v78 == -4096 || *v78 == -8192) );
            }
            while ( v78 != (__int64 *)((char *)v79 + v30) );
            v99 = v125;
            v95 = (unsigned int)v125;
            v103 = (unsigned __int64)v124;
            v109 = &v124[16 * (unsigned int)v125];
          }
        }
      }
    }
    --v93;
  }
  while ( v93 );
  if ( v109 == (char *)v103 )
  {
LABEL_51:
    v101 = v106;
    goto LABEL_52;
  }
LABEL_36:
  v44 = (unsigned __int8 **)v103;
  v45 = 0;
  while ( 1 )
  {
    v46 = *v44;
    v47 = 32LL * (*((_DWORD *)*v44 + 1) & 0x7FFFFFF);
    if ( ((*v44)[7] & 0x40) != 0 )
    {
      v48 = (unsigned __int8 *)*((_QWORD *)v46 - 1);
      v49 = &v48[v47];
    }
    else
    {
      v48 = &v46[-v47];
      v49 = *v44;
    }
    v50 = v49 - v48;
    v111 = (unsigned __int8 **)v113;
    v112 = 0x400000000LL;
    v51 = v50 >> 5;
    v52 = v50 >> 5;
    if ( (unsigned __int64)v50 > 0x80 )
    {
      v94 = v50 >> 5;
      v96 = v50;
      v100 = v48;
      v105 = v50 >> 5;
      sub_C8D5F0((__int64)&v111, v113, v51, 8u, v50, v30);
      v55 = v111;
      v54 = v112;
      LODWORD(v51) = v105;
      v48 = v100;
      v50 = v96;
      v52 = v94;
      v53 = &v111[(unsigned int)v112];
    }
    else
    {
      v53 = (unsigned __int8 **)v113;
      v54 = 0;
      v55 = (unsigned __int8 **)v113;
    }
    if ( v50 > 0 )
    {
      v56 = 0;
      do
      {
        v53[v56 / 8] = *(unsigned __int8 **)&v48[4 * v56];
        v56 += 8LL;
        --v52;
      }
      while ( v52 );
      v55 = v111;
      v54 = v112;
    }
    LODWORD(v112) = v51 + v54;
    v58 = sub_DFCEF0(&v110, v46, v55, (unsigned int)(v51 + v54), 1);
    if ( v111 != (unsigned __int8 **)v113 )
    {
      v104 = v57;
      _libc_free((unsigned __int64)v111);
      v57 = v104;
    }
    if ( v57 == 1 )
      break;
    v59 = __OFADD__(v58, v45);
    v45 += v58;
    if ( v59 )
    {
      if ( v58 > 0 )
        break;
      v45 = 0x8000000000000000LL;
    }
    else if ( v97 < v45 )
    {
      break;
    }
    v44 += 2;
    if ( v109 == (char *)v44 )
      goto LABEL_51;
  }
LABEL_52:
  sub_DFE7B0(&v110);
  v106 = v101;
LABEL_53:
  if ( v124 != v126 )
    _libc_free((unsigned __int64)v124);
  if ( (v121 & 1) == 0 )
    sub_C7D6A0((__int64)v122, 16LL * v123, 8);
  if ( (_BYTE *)v118[0] != v119 )
    _libc_free(v118[0]);
  if ( (v115 & 1) == 0 )
    sub_C7D6A0((__int64)v116, 16LL * v117, 8);
  return v106;
}
