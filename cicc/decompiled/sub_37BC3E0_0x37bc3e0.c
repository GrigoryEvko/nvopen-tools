// Function: sub_37BC3E0
// Address: 0x37bc3e0
//
void __fastcall sub_37BC3E0(__int64 *a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 v4; // rdi
  __int64 v5; // r8
  __int64 v6; // r9
  bool v7; // zf
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // r15
  __int64 v12; // rdx
  _QWORD *v13; // r14
  unsigned int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned int v21; // eax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 *v24; // r14
  __int64 v25; // rax
  __int64 *v26; // r13
  __int64 *v27; // r12
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rbx
  char *v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // r14
  __int64 *k; // r15
  __int64 v38; // rbx
  __int64 *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  char v42; // dl
  __int64 v43; // r10
  __int64 v44; // rdi
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  unsigned __int64 v47; // rdx
  int v48; // eax
  __int64 *v49; // rdx
  __int64 v50; // rdi
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  __int64 *v53; // rax
  unsigned __int64 v54; // r8
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 *v57; // rax
  __int64 v58; // rdi
  __int64 *v59; // rax
  _BYTE *v60; // rax
  __int64 v61; // r10
  __int64 v62; // r8
  __int64 v63; // rax
  __int64 v64; // r9
  __int64 v65; // rcx
  __int64 j; // rdx
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // r11
  __int64 v70; // rax
  unsigned int v71; // r14d
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rdx
  unsigned int i; // eax
  __int64 v78; // r13
  unsigned __int64 v79; // rbx
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  __int64 *v82; // rax
  __int64 v83; // rdi
  __int64 *v84; // rdx
  __int64 v85; // rcx
  __int64 *v86; // rax
  _QWORD *v87; // rax
  unsigned int v88; // eax
  __int64 v89; // rdx
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // rsi
  unsigned int v93; // eax
  __int64 v94; // rcx
  __int64 v95; // rsi
  unsigned __int64 v96; // rdx
  unsigned int v97; // eax
  unsigned int v98; // [rsp+20h] [rbp-4C0h]
  __int64 v99; // [rsp+20h] [rbp-4C0h]
  unsigned int v100; // [rsp+30h] [rbp-4B0h]
  unsigned int v101; // [rsp+30h] [rbp-4B0h]
  __int64 v102; // [rsp+30h] [rbp-4B0h]
  unsigned __int64 v103; // [rsp+30h] [rbp-4B0h]
  unsigned int v104; // [rsp+30h] [rbp-4B0h]
  unsigned int v105; // [rsp+38h] [rbp-4A8h]
  __int64 v106; // [rsp+38h] [rbp-4A8h]
  __int64 v107; // [rsp+40h] [rbp-4A0h] BYREF
  char *v108; // [rsp+48h] [rbp-498h]
  __int64 v109; // [rsp+50h] [rbp-490h]
  int v110; // [rsp+58h] [rbp-488h]
  char v111; // [rsp+5Ch] [rbp-484h]
  char v112; // [rsp+60h] [rbp-480h] BYREF
  __int64 v113; // [rsp+E0h] [rbp-400h] BYREF
  __int64 *v114; // [rsp+E8h] [rbp-3F8h]
  __int64 v115; // [rsp+F0h] [rbp-3F0h]
  int v116; // [rsp+F8h] [rbp-3E8h]
  char v117; // [rsp+FCh] [rbp-3E4h]
  char v118; // [rsp+100h] [rbp-3E0h] BYREF
  _BYTE *v119; // [rsp+180h] [rbp-360h] BYREF
  __int64 v120; // [rsp+188h] [rbp-358h]
  _BYTE v121[256]; // [rsp+190h] [rbp-350h] BYREF
  _BYTE *v122; // [rsp+290h] [rbp-250h] BYREF
  __int64 v123; // [rsp+298h] [rbp-248h]
  _BYTE v124[576]; // [rsp+2A0h] [rbp-240h] BYREF

  v3 = a1;
  v4 = *a1;
  v122 = v124;
  v123 = 0x2000000000LL;
  sub_2E6D080(v4);
  v7 = *((_BYTE *)v3 + 9) == 0;
  v120 = 0x2000000000LL;
  v119 = v121;
  v108 = &v112;
  v107 = 0;
  v109 = 16;
  v110 = 0;
  v111 = 1;
  v113 = 0;
  v114 = (__int64 *)&v118;
  v115 = 16;
  v116 = 0;
  v117 = 1;
  if ( v7 )
    goto LABEL_2;
  v107 = 1;
  v88 = *(_DWORD *)(v3[2] + 20) - *(_DWORD *)(v3[2] + 24);
  v89 = 1;
  if ( v88 > 0x10 )
  {
    v94 = 2863311531LL;
    v95 = 128;
    v96 = v88 / 3uLL;
    v97 = v88 + v96 - 1;
    if ( v97 )
    {
      _BitScanReverse(&v97, v97);
      v94 = 33 - (v97 ^ 0x1F);
      v95 = (unsigned int)(1 << (33 - (v97 ^ 0x1F)));
      if ( (unsigned int)v95 < 0x80 )
        v95 = 128;
    }
    sub_C8CB60((__int64)&v107, v95, v96, v94, v5, v6);
    v89 = v113 + 1;
    v88 = *(_DWORD *)(v3[2] + 20) - *(_DWORD *)(v3[2] + 24);
  }
  v113 = v89;
  if ( !v88 )
    goto LABEL_2;
  if ( v117 )
  {
    v90 = v88 - 1;
    if ( v88 > (unsigned int)v115 )
    {
LABEL_120:
      v91 = 2863311531LL;
      v92 = 128;
      v93 = v90 + v88 / 3;
      if ( v93 )
      {
        _BitScanReverse(&v93, v93);
        v91 = 33 - (v93 ^ 0x1F);
        v92 = (unsigned int)(1 << (33 - (v93 ^ 0x1F)));
        if ( (unsigned int)v92 < 0x80 )
          v92 = 128;
      }
      sub_C8CB60((__int64)&v113, v92, v90, v91, v5, v6);
    }
  }
  else
  {
    v90 = v88 - 1;
    if ( 4 * (int)v90 >= (unsigned int)(3 * v115) )
      goto LABEL_120;
  }
LABEL_2:
  v8 = v3[3];
  v9 = *(_QWORD **)(v8 + 8);
  if ( *(_BYTE *)(v8 + 28) )
    v10 = *(unsigned int *)(v8 + 20);
  else
    v10 = *(unsigned int *)(v8 + 16);
  v11 = &v9[v10];
  if ( v9 == v11 )
    goto LABEL_7;
  while ( 1 )
  {
    v12 = *v9;
    v13 = v9;
    if ( *v9 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v11 == ++v9 )
      goto LABEL_7;
  }
  if ( v11 != v9 )
  {
    v106 = a2;
    v75 = *v3;
    if ( !v12 )
      goto LABEL_110;
LABEL_94:
    v76 = (unsigned int)(*(_DWORD *)(v12 + 24) + 1);
    for ( i = v76; ; i = 0 )
    {
      if ( i >= *(_DWORD *)(v75 + 32) )
        goto LABEL_104;
      v78 = *(_QWORD *)(*(_QWORD *)(v75 + 24) + 8 * v76);
      if ( !v78 )
        goto LABEL_104;
      v79 = ((unsigned __int64)*(unsigned int *)(v78 + 72) << 32) | *(unsigned int *)(v78 + 16);
      v80 = (unsigned int)v123;
      v81 = (unsigned int)v123 + 1LL;
      if ( v81 > HIDWORD(v123) )
      {
        sub_C8D5F0((__int64)&v122, v124, v81, 0x10u, v5, v6);
        v80 = (unsigned int)v123;
      }
      v82 = (__int64 *)&v122[16 * v80];
      *v82 = v78;
      v83 = (__int64)v122;
      v82[1] = v79;
      LODWORD(v123) = v123 + 1;
      sub_37B6780(
        v83,
        ((16LL * (unsigned int)v123) >> 4) - 1,
        0,
        *(_QWORD *)(v83 + 16LL * (unsigned int)v123 - 16),
        *(_QWORD *)(v83 + 16LL * (unsigned int)v123 - 8));
      if ( !v117 )
        goto LABEL_111;
      v86 = v114;
      v85 = HIDWORD(v115);
      v84 = &v114[HIDWORD(v115)];
      if ( v114 == v84 )
        break;
      while ( v78 != *v86 )
      {
        if ( v84 == ++v86 )
          goto LABEL_112;
      }
LABEL_104:
      v87 = v13 + 1;
      if ( v13 + 1 == v11 )
        goto LABEL_107;
      while ( 1 )
      {
        v12 = *v87;
        v13 = v87;
        if ( *v87 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v11 == ++v87 )
          goto LABEL_107;
      }
      if ( v11 == v87 )
      {
LABEL_107:
        a2 = v106;
        goto LABEL_7;
      }
      v75 = *v3;
      if ( v12 )
        goto LABEL_94;
LABEL_110:
      v76 = 0;
    }
LABEL_112:
    if ( HIDWORD(v115) < (unsigned int)v115 )
    {
      ++HIDWORD(v115);
      *v84 = v78;
      ++v113;
      goto LABEL_104;
    }
LABEL_111:
    sub_C8CC70((__int64)&v113, v78, (__int64)v84, v85, v5, v6);
    goto LABEL_104;
  }
LABEL_7:
  v14 = v123;
  if ( !(_DWORD)v123 )
    goto LABEL_34;
  do
  {
    v15 = (__int64)v122;
    v16 = v14;
    v17 = *(_QWORD *)v122;
    v18 = 16LL * v14;
    v105 = *((_DWORD *)v122 + 2);
    if ( v16 != 1 )
    {
      v60 = &v122[v18];
      v61 = *((_QWORD *)v60 - 2);
      v62 = *((_QWORD *)v60 - 1);
      *((_QWORD *)v60 - 2) = v17;
      v60 -= 16;
      *((_DWORD *)v60 + 2) = *(_DWORD *)(v15 + 8);
      *((_DWORD *)v60 + 3) = *(_DWORD *)(v15 + 12);
      v63 = (__int64)&v60[-v15];
      v64 = v63 >> 4;
      v65 = ((v63 >> 4) - 1) / 2;
      if ( v63 <= 32 )
      {
        v67 = 0;
      }
      else
      {
        for ( j = 0; ; j = v67 )
        {
          v67 = 2 * (j + 1);
          v68 = 32 * (j + 1);
          v69 = v15 + v68 - 16;
          v70 = v15 + v68;
          v71 = *(_DWORD *)(v69 + 8);
          if ( *(_DWORD *)(v70 + 8) < v71
            || *(_DWORD *)(v70 + 8) == v71 && *(_DWORD *)(v70 + 12) < *(_DWORD *)(v69 + 12) )
          {
            --v67;
            v70 = v15 + 16 * v67;
          }
          v72 = v15 + 16 * j;
          *(_QWORD *)v72 = *(_QWORD *)v70;
          *(_DWORD *)(v72 + 8) = *(_DWORD *)(v70 + 8);
          *(_DWORD *)(v72 + 12) = *(_DWORD *)(v70 + 12);
          if ( v65 <= v67 )
            break;
        }
      }
      if ( (v64 & 1) == 0 && (v64 - 2) / 2 == v67 )
      {
        v73 = v15 + 32 * (v67 + 1) - 16;
        v74 = v15 + 16 * v67;
        *(_QWORD *)v74 = *(_QWORD *)v73;
        *(_DWORD *)(v74 + 8) = *(_DWORD *)(v73 + 8);
        v67 = 2 * (v67 + 1) - 1;
        *(_DWORD *)(v74 + 12) = *(_DWORD *)(v73 + 12);
      }
      sub_37B6780(v15, v67, 0, v61, v62);
    }
    v19 = (unsigned int)v120;
    LODWORD(v123) = v123 - 1;
    v20 = (unsigned int)v120 + 1LL;
    if ( v20 > HIDWORD(v120) )
    {
      sub_C8D5F0((__int64)&v119, v121, v20, 8u, v5, v6);
      v19 = (unsigned int)v120;
    }
    *(_QWORD *)&v119[8 * v19] = v17;
    v7 = (_DWORD)v120 == -1;
    v21 = v120 + 1;
    LODWORD(v120) = v120 + 1;
    if ( v7 )
      goto LABEL_33;
    do
    {
      v22 = v21;
      v23 = *(_QWORD *)&v119[8 * v21 - 8];
      LODWORD(v120) = v21 - 1;
      v6 = *(_QWORD *)(*(_QWORD *)v23 + 112LL);
      v24 = (__int64 *)(v6 + 8LL * *(unsigned int *)(*(_QWORD *)v23 + 120LL));
      if ( (__int64 *)v6 == v24 )
        goto LABEL_25;
      v25 = a2;
      v26 = v3;
      v27 = *(__int64 **)(*(_QWORD *)v23 + 112LL);
      v28 = v25;
      do
      {
        while ( 1 )
        {
          v29 = *v27;
          v30 = *v26;
          if ( *v27 )
          {
            v22 = (unsigned int)(*(_DWORD *)(v29 + 24) + 1);
            v31 = *(_DWORD *)(v29 + 24) + 1;
          }
          else
          {
            v22 = 0;
            v31 = 0;
          }
          if ( v31 >= *(_DWORD *)(v30 + 32) )
            BUG();
          v32 = *(_QWORD *)(*(_QWORD *)(v30 + 24) + 8 * v22);
          v5 = *(unsigned int *)(v32 + 16);
          if ( v105 < (unsigned int)v5 )
            goto LABEL_23;
          if ( !v111 )
            goto LABEL_51;
          v33 = v108;
          v22 = HIDWORD(v109);
          v30 = (__int64)&v108[8 * HIDWORD(v109)];
          if ( v108 != (char *)v30 )
          {
            while ( v32 != *(_QWORD *)v33 )
            {
              v33 += 8;
              if ( (char *)v30 == v33 )
                goto LABEL_68;
            }
            goto LABEL_23;
          }
LABEL_68:
          if ( HIDWORD(v109) < (unsigned int)v109 )
          {
            v22 = (unsigned int)++HIDWORD(v109);
            *(_QWORD *)v30 = v32;
            ++v107;
          }
          else
          {
LABEL_51:
            v100 = *(_DWORD *)(v32 + 16);
            sub_C8CC70((__int64)&v107, v32, v30, v22, v5, v6);
            v5 = v100;
            if ( !v42 )
              goto LABEL_23;
          }
          v43 = *(_QWORD *)v32;
          if ( !*((_BYTE *)v26 + 9) )
            goto LABEL_58;
          v44 = v26[2];
          if ( *(_BYTE *)(v44 + 28) )
          {
            v45 = *(_QWORD **)(v44 + 8);
            v46 = &v45[*(unsigned int *)(v44 + 20)];
            if ( v45 == v46 )
              goto LABEL_23;
            while ( v43 != *v45 )
            {
              if ( v46 == ++v45 )
                goto LABEL_23;
            }
LABEL_58:
            v47 = *(unsigned int *)(v28 + 8);
            v22 = *(unsigned int *)(v28 + 12);
            v48 = *(_DWORD *)(v28 + 8);
            if ( v47 < v22 )
              goto LABEL_59;
            goto LABEL_77;
          }
          v98 = v5;
          v102 = *(_QWORD *)v32;
          v59 = sub_C8CA60(v44, *(_QWORD *)v32);
          v43 = v102;
          v5 = v98;
          if ( !v59 )
            goto LABEL_23;
          v47 = *(unsigned int *)(v28 + 8);
          v22 = *(unsigned int *)(v28 + 12);
          v48 = *(_DWORD *)(v28 + 8);
          if ( v47 < v22 )
          {
LABEL_59:
            v22 = *(_QWORD *)v28;
            v49 = (__int64 *)(*(_QWORD *)v28 + 8 * v47);
            if ( v49 )
            {
              *v49 = v43;
              v48 = *(_DWORD *)(v28 + 8);
            }
            *(_DWORD *)(v28 + 8) = v48 + 1;
            goto LABEL_62;
          }
LABEL_77:
          if ( v22 < v47 + 1 )
          {
            v99 = v43;
            v104 = v5;
            sub_C8D5F0(v28, (const void *)(v28 + 16), v47 + 1, 8u, v5, v6);
            v47 = *(unsigned int *)(v28 + 8);
            v43 = v99;
            v5 = v104;
          }
          *(_QWORD *)(*(_QWORD *)v28 + 8 * v47) = v43;
          ++*(_DWORD *)(v28 + 8);
LABEL_62:
          v50 = v26[3];
          if ( *(_BYTE *)(v50 + 28) )
            break;
          v101 = v5;
          v53 = sub_C8CA60(v50, v43);
          v5 = v101;
          if ( !v53 )
            goto LABEL_71;
LABEL_23:
          if ( v24 == ++v27 )
            goto LABEL_24;
        }
        v51 = *(_QWORD **)(v50 + 8);
        v52 = &v51[*(unsigned int *)(v50 + 20)];
        if ( v51 != v52 )
        {
          while ( v43 != *v51 )
          {
            if ( v52 == ++v51 )
              goto LABEL_71;
          }
          goto LABEL_23;
        }
LABEL_71:
        v54 = ((unsigned __int64)*(unsigned int *)(v32 + 72) << 32) | v5;
        v55 = (unsigned int)v123;
        v56 = (unsigned int)v123 + 1LL;
        if ( v56 > HIDWORD(v123) )
        {
          v103 = v54;
          sub_C8D5F0((__int64)&v122, v124, v56, 0x10u, v54, v6);
          v55 = (unsigned int)v123;
          v54 = v103;
        }
        v57 = (__int64 *)&v122[16 * v55];
        ++v27;
        *v57 = v32;
        v58 = (__int64)v122;
        v57[1] = v54;
        LODWORD(v123) = v123 + 1;
        sub_37B6780(
          v58,
          ((16LL * (unsigned int)v123) >> 4) - 1,
          0,
          *(_QWORD *)(v58 + 16LL * (unsigned int)v123 - 16),
          *(_QWORD *)(v58 + 16LL * (unsigned int)v123 - 8));
      }
      while ( v24 != v27 );
LABEL_24:
      v3 = v26;
      a2 = v28;
LABEL_25:
      v34 = *(__int64 **)(v23 + 24);
      v35 = *(unsigned int *)(v23 + 32);
      v36 = &v34[v35];
      for ( k = v34; v36 != k; LODWORD(v120) = v120 + 1 )
      {
LABEL_26:
        v38 = *k;
        if ( !v117 )
          goto LABEL_43;
        v39 = v114;
        v22 = HIDWORD(v115);
        v35 = (__int64)&v114[HIDWORD(v115)];
        if ( v114 != (__int64 *)v35 )
        {
          while ( v38 != *v39 )
          {
            if ( (__int64 *)v35 == ++v39 )
              goto LABEL_48;
          }
LABEL_31:
          if ( v36 == ++k )
            break;
          goto LABEL_26;
        }
LABEL_48:
        if ( HIDWORD(v115) < (unsigned int)v115 )
        {
          ++HIDWORD(v115);
          *(_QWORD *)v35 = v38;
          ++v113;
        }
        else
        {
LABEL_43:
          sub_C8CC70((__int64)&v113, *k, v35, v22, v5, v6);
          if ( !(_BYTE)v35 )
            goto LABEL_31;
        }
        v40 = (unsigned int)v120;
        v22 = HIDWORD(v120);
        v41 = (unsigned int)v120 + 1LL;
        if ( v41 > HIDWORD(v120) )
        {
          sub_C8D5F0((__int64)&v119, v121, v41, 8u, v5, v6);
          v40 = (unsigned int)v120;
        }
        v35 = (__int64)v119;
        ++k;
        *(_QWORD *)&v119[8 * v40] = v38;
      }
      v21 = v120;
    }
    while ( (_DWORD)v120 );
LABEL_33:
    v14 = v123;
  }
  while ( (_DWORD)v123 );
LABEL_34:
  if ( !v117 )
    _libc_free((unsigned __int64)v114);
  if ( !v111 )
    _libc_free((unsigned __int64)v108);
  if ( v119 != v121 )
    _libc_free((unsigned __int64)v119);
  if ( v122 != v124 )
    _libc_free((unsigned __int64)v122);
}
