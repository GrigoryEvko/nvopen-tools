// Function: sub_2B7CA70
// Address: 0x2b7ca70
//
__int64 ***__fastcall sub_2B7CA70(
        __int64 a1,
        __int64 a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 **a8,
        int a9,
        __int64 a10)
{
  _BYTE *v12; // rdx
  signed int v13; // esi
  __int64 v14; // rcx
  __int64 ***v15; // r13
  int *v16; // rcx
  unsigned int v17; // r14d
  __int64 v18; // r10
  __int64 v19; // r9
  int v20; // eax
  _DWORD *v21; // rsi
  unsigned __int64 v22; // r8
  __int64 v23; // r9
  int *v24; // rcx
  char v25; // r10
  __int64 v26; // r8
  int *v27; // r14
  __int64 v28; // rcx
  unsigned __int64 v29; // r8
  __int64 v30; // r9
  char v31; // r10
  __int64 v32; // rdx
  __int64 ***v33; // rax
  char v34; // bl
  unsigned __int64 v35; // r8
  int *v36; // r11
  __int64 v37; // rax
  int *v38; // rdx
  int v39; // ecx
  __int64 *v40; // rdi
  __int64 v41; // rbx
  unsigned int v42; // esi
  unsigned int v43; // edi
  _QWORD *v44; // rax
  __int64 ***v45; // r9
  int v46; // ebx
  _DWORD *v47; // rsi
  unsigned __int64 v48; // r8
  __int64 v49; // r9
  int *v50; // rax
  __int64 ***v51; // rax
  char v52; // bl
  unsigned __int64 v53; // r8
  __int64 v54; // rax
  int v55; // edx
  int *v56; // rdi
  int v57; // esi
  unsigned __int64 v58; // rdx
  __int64 ***v59; // rax
  void *v60; // r14
  __int64 ***v61; // r14
  __int64 v62; // r8
  int *v63; // r12
  unsigned __int64 v64; // r8
  __int64 v65; // r9
  size_t v66; // r11
  char *v67; // r10
  char *v68; // rax
  char *v69; // rcx
  __int64 v70; // rsi
  char v71; // r10
  unsigned __int64 v72; // r8
  _DWORD *v73; // r9
  __int64 v74; // rax
  int v75; // edx
  _DWORD *v76; // rsi
  unsigned __int64 v77; // r12
  unsigned __int64 v78; // r12
  __int64 ***v80; // rax
  int v81; // esi
  _DWORD *v82; // rsi
  _DWORD *v83; // rax
  __int64 v84; // rsi
  unsigned __int64 v85; // rax
  __int64 v86; // rdi
  int v87; // edx
  __int64 ***v88; // rax
  __int64 ***v89; // rax
  char v90; // al
  __int64 v91; // [rsp+8h] [rbp-F8h]
  __int64 v92; // [rsp+8h] [rbp-F8h]
  int n; // [rsp+10h] [rbp-F0h]
  size_t na; // [rsp+10h] [rbp-F0h]
  char v95; // [rsp+18h] [rbp-E8h]
  __int64 ***v96; // [rsp+18h] [rbp-E8h]
  int v97; // [rsp+18h] [rbp-E8h]
  int *v98; // [rsp+20h] [rbp-E0h]
  char v99; // [rsp+20h] [rbp-E0h]
  int *v100; // [rsp+20h] [rbp-E0h]
  __int64 ***v101; // [rsp+20h] [rbp-E0h]
  __int64 v102; // [rsp+20h] [rbp-E0h]
  int v103; // [rsp+20h] [rbp-E0h]
  char *v104; // [rsp+20h] [rbp-E0h]
  char v105; // [rsp+20h] [rbp-E0h]
  __int64 v107; // [rsp+38h] [rbp-C8h]
  int *v108; // [rsp+38h] [rbp-C8h]
  int *v109; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v110; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 v111; // [rsp+48h] [rbp-B8h] BYREF
  _DWORD *v112; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v113; // [rsp+58h] [rbp-A8h]
  _BYTE v114[48]; // [rsp+60h] [rbp-A0h] BYREF
  void *v115; // [rsp+90h] [rbp-70h] BYREF
  __int64 v116; // [rsp+98h] [rbp-68h]
  _BYTE s[96]; // [rsp+A0h] [rbp-60h] BYREF

  v112 = v114;
  v113 = 0xC00000000LL;
  if ( *(_DWORD *)(a1 + 16) )
  {
    sub_2B0D670((__int64)&v112, a1 + 8, (__int64)a3, a4, a5, a6);
    v14 = (unsigned int)v113;
    v12 = v112;
    v13 = v113;
  }
  else
  {
    v12 = v114;
    v13 = 0;
    v14 = 0;
  }
  sub_2B23C00((__int64 *)&v110, v13, (__int64)v12, v14, 2);
  sub_2B25A00(&v111, a3, (unsigned __int64 *)&v110);
  if ( (v111 & 1) != 0 )
  {
    if ( (~(-1LL << (v111 >> 58)) & (v111 >> 1)) == (1LL << (v111 >> 58)) - 1 )
      goto LABEL_5;
LABEL_47:
    v45 = *(__int64 ****)a1;
    v46 = v113;
    if ( (_DWORD)v113 != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 32LL) )
    {
      v47 = &v112[(unsigned int)v113];
      v100 = v112;
      if ( v47 != sub_2B09020(v112, (__int64)v47, v113) )
      {
        v52 = 1;
        v45 = sub_2B7C3E0(*a8, v49, 0, (__int64)v100, v48);
LABEL_57:
        v102 = (__int64)v45;
        sub_2B25530(&v115, (__int64)a3, (unsigned __int64 *)&v110);
        v53 = (unsigned int)v113;
        if ( (_DWORD)v113 )
        {
          v54 = 0;
          do
          {
            while ( 1 )
            {
              v55 = v54;
              v56 = &v112[v54];
              v57 = *v56;
              if ( *v56 != -1 )
                break;
              if ( ((unsigned __int8)v115 & 1) != 0 )
                v58 = ((((unsigned __int64)v115 >> 1) & ~(-1LL << ((unsigned __int64)v115 >> 58))) >> v54) & 1;
              else
                v58 = (*(_QWORD *)(*(_QWORD *)v115 + 8LL * ((unsigned int)v54 >> 6)) >> v54) & 1LL;
              if ( !(_BYTE)v58 )
                v57 = v54;
              ++v54;
              *v56 = v57;
              if ( (unsigned int)v53 == v54 )
                goto LABEL_68;
            }
            if ( !v52 )
              v55 = *v56;
            ++v54;
            *v56 = v53 + v55;
          }
          while ( (unsigned int)v53 != v54 );
LABEL_68:
          v53 = (unsigned int)v113;
        }
        v59 = sub_2B7C3E0(*(__int64 **)(a10 + 8), *(_QWORD *)(*(_QWORD *)a10 - 96LL), v102, (__int64)v112, v53);
        v60 = v115;
        v15 = v59;
        if ( ((unsigned __int8)v115 & 1) == 0 && v115 )
        {
          if ( *(void **)v115 != (char *)v115 + 16 )
            _libc_free(*(_QWORD *)v115);
          j_j___libc_free_0((unsigned __int64)v60);
        }
        v41 = a1 + 72;
        goto LABEL_72;
      }
      v115 = s;
      v92 = v49;
      v116 = 0xC00000000LL;
      sub_11B1960((__int64)&v115, v48, -1, (__int64)v100, v48, v49);
      if ( v46 )
      {
        v50 = v100;
        do
        {
          if ( *v50 != -1 )
            *((_DWORD *)v115 + *v50) = *v50;
          ++v50;
        }
        while ( &v100[v46 - 1 + 1] != v50 );
      }
      v51 = sub_2B7C3E0(*a8, v92, 0, (__int64)v115, (unsigned int)v116);
      v45 = v51;
      if ( v115 != s )
      {
        v101 = v51;
        _libc_free((unsigned __int64)v115);
        v45 = v101;
      }
    }
    v52 = 0;
    goto LABEL_57;
  }
  v42 = *(_DWORD *)(v111 + 64);
  v43 = v42 >> 6;
  if ( v42 >> 6 )
  {
    v44 = *(_QWORD **)v111;
    while ( *v44 == -1 )
    {
      if ( (_QWORD *)(*(_QWORD *)v111 + 8LL * (v43 - 1) + 8) == ++v44 )
        goto LABEL_118;
    }
    goto LABEL_47;
  }
LABEL_118:
  v81 = v42 & 0x3F;
  if ( v81 && *(_QWORD *)(*(_QWORD *)v111 + 8LL * v43) != (1LL << v81) - 1 )
    goto LABEL_47;
LABEL_5:
  v15 = *(__int64 ****)a1;
  v16 = v112;
  v17 = v113;
  v18 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 32LL);
  if ( a2 == 1 )
  {
    if ( (_DWORD)v113 == (_DWORD)v18
      || (v82 = &v112[(unsigned int)v113],
          v108 = v112,
          v83 = sub_2B09020(v112, (__int64)v82, v113),
          v16 = v108,
          v82 == v83) )
    {
      if ( v17 != v18 || (v109 = v16, v90 = sub_B4ED80(v16, v17, v17), v16 = v109, !v90) )
        v15 = sub_2B7C3E0(*(__int64 **)(a10 + 8), (__int64)v15, 0, (__int64)v16, v17);
    }
    else
    {
      v15 = sub_2B7C3E0(*a8, (__int64)v15, 0, (__int64)v108, v17);
    }
    goto LABEL_102;
  }
  v19 = *(_QWORD *)(a1 + 72);
  v20 = *(_DWORD *)(*(_QWORD *)(v19 + 8) + 32LL);
  if ( v20 == (_DWORD)v18 )
  {
    v84 = *(_QWORD *)(a1 + 80);
    if ( (_DWORD)v113 )
    {
      v85 = 0;
      v86 = 4LL * (unsigned int)v113;
      do
      {
        v87 = *(_DWORD *)(v84 + v85);
        if ( v87 != -1 )
          v112[v85 / 4] = v18 + v87;
        v85 += 4LL;
      }
      while ( v86 != v85 );
      v15 = *(__int64 ****)a1;
      v19 = *(_QWORD *)(a1 + 72);
      v16 = v112;
      v35 = (unsigned int)v113;
      v40 = *(__int64 **)(a10 + 8);
      if ( !*(_QWORD *)a1 )
        v15 = *(__int64 ****)(*(_QWORD *)a10 - 96LL);
    }
    else
    {
      v40 = *(__int64 **)(a10 + 8);
      v35 = 0;
    }
    goto LABEL_42;
  }
  if ( (_DWORD)v113 == (_DWORD)v18 )
    goto LABEL_16;
  v21 = &v112[(unsigned int)v113];
  v98 = v112;
  if ( v21 == sub_2B09020(v112, (__int64)v21, v113) )
  {
    v115 = s;
    v116 = 0xC00000000LL;
    sub_11B1960((__int64)&v115, v22, -1, (__int64)v98, v22, v23);
    v24 = v98;
    if ( v17 )
    {
      do
      {
        if ( *v24 != -1 )
          *((_DWORD *)v115 + *v24) = *v24;
        ++v24;
      }
      while ( &v98[v17] != v24 );
    }
    v15 = sub_2B7C3E0(*a8, (__int64)v15, 0, (__int64)v115, (unsigned int)v116);
    if ( v115 != s )
      _libc_free((unsigned __int64)v115);
    v19 = *(_QWORD *)(a1 + 72);
    v20 = *(_DWORD *)(*(_QWORD *)(v19 + 8) + 32LL);
LABEL_16:
    v25 = 0;
    goto LABEL_17;
  }
  v89 = sub_2B7C3E0(*a8, (__int64)v15, 0, (__int64)v98, v22);
  v19 = *(_QWORD *)(a1 + 72);
  v25 = 1;
  v15 = v89;
  v20 = *(_DWORD *)(*(_QWORD *)(v19 + 8) + 32LL);
LABEL_17:
  v26 = *(unsigned int *)(a1 + 88);
  v27 = *(int **)(a1 + 80);
  if ( (_DWORD)v26 != v20 )
  {
    if ( &v27[v26] != sub_2B09020(*(_DWORD **)(a1 + 80), (__int64)&v27[v26], v26) )
    {
      v105 = v31;
      v34 = 1;
      v88 = sub_2B7C3E0(*a8, v30, 0, (__int64)v27, v29);
      v27 = *(int **)(a1 + 80);
      v25 = v105;
      v19 = (__int64)v88;
      goto LABEL_28;
    }
    v115 = s;
    v91 = v30;
    v95 = v31;
    n = v29;
    v116 = 0xC00000000LL;
    sub_11B1960((__int64)&v115, v29, -1, v28, v29, v30);
    if ( n )
    {
      v32 = (__int64)&v27[n - 1 + 1];
      do
      {
        if ( *v27 != -1 )
          *((_DWORD *)v115 + *v27) = *v27;
        ++v27;
      }
      while ( (int *)v32 != v27 );
    }
    v33 = sub_2B7C3E0(*a8, v91, 0, (__int64)v115, (unsigned int)v116);
    v25 = v95;
    v19 = (__int64)v33;
    if ( v115 != s )
    {
      v99 = v95;
      v96 = v33;
      _libc_free((unsigned __int64)v115);
      v19 = (__int64)v96;
      v25 = v99;
    }
    v27 = *(int **)(a1 + 80);
  }
  v34 = 0;
LABEL_28:
  v35 = (unsigned int)v113;
  v36 = v112;
  if ( !(_DWORD)v113 )
    goto LABEL_39;
  v37 = 0;
  do
  {
    while ( 1 )
    {
      v38 = &v36[v37];
      v39 = v37;
      if ( *v38 != -1 )
      {
        if ( v25 )
        {
          *v38 = v37;
          v36 = v112;
        }
        goto LABEL_32;
      }
      if ( v27[v37] != -1 )
        break;
LABEL_32:
      if ( (unsigned int)v35 == ++v37 )
        goto LABEL_38;
    }
    if ( !v34 )
      v39 = v27[v37];
    ++v37;
    *v38 = v35 + v39;
    v36 = v112;
  }
  while ( (unsigned int)v35 != v37 );
LABEL_38:
  v35 = (unsigned int)v113;
LABEL_39:
  v40 = *(__int64 **)(a10 + 8);
  if ( !v15 )
    v15 = *(__int64 ****)(*(_QWORD *)a10 - 96LL);
  v16 = v36;
LABEL_42:
  v41 = a1 + 144;
  v15 = sub_2B7C3E0(v40, (__int64)v15, v19, (__int64)v16, v35);
LABEL_72:
  v107 = a1 + 72 * a2;
  if ( v107 == v41 )
    goto LABEL_102;
  while ( 2 )
  {
    v61 = *(__int64 ****)v41;
    v62 = *(unsigned int *)(v41 + 16);
    v63 = *(int **)(v41 + 8);
    if ( (_DWORD)v62 == *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v41 + 8LL) + 32LL) )
    {
LABEL_88:
      v71 = 0;
    }
    else
    {
      if ( &v63[v62] == sub_2B09020(*(_DWORD **)(v41 + 8), (__int64)&v63[v62], v62) )
      {
        v67 = s;
        v115 = s;
        v116 = 0xC00000000LL;
        if ( v64 > 0xC )
        {
          na = v66;
          v97 = v64;
          sub_C8D5F0((__int64)&v115, s, v64, 4u, v64, v65);
          v67 = s;
          LODWORD(v65) = v97;
          v69 = (char *)v115 + na;
          if ( v115 != (char *)v115 + na )
          {
            memset(v115, 255, na);
            v69 = (char *)v115;
            v67 = s;
            LODWORD(v65) = v97;
          }
          LODWORD(v116) = v65;
        }
        else
        {
          if ( v64 && v66 )
          {
            v103 = v64;
            v68 = (char *)memset(s, 255, v66);
            LODWORD(v65) = v103;
            v67 = v68;
          }
          LODWORD(v116) = v65;
          v69 = v67;
        }
        if ( (_DWORD)v65 )
        {
          v70 = (__int64)&v63[(unsigned int)(v65 - 1) + 1];
          do
          {
            if ( *v63 != -1 )
            {
              *(_DWORD *)&v69[4 * *v63] = *v63;
              v69 = (char *)v115;
            }
            ++v63;
          }
          while ( (int *)v70 != v63 );
        }
        v104 = v67;
        v61 = sub_2B7C3E0(*a8, (__int64)v61, 0, (__int64)v69, (unsigned int)v116);
        if ( v115 != v104 )
          _libc_free((unsigned __int64)v115);
        v63 = *(int **)(v41 + 8);
        goto LABEL_88;
      }
      v80 = sub_2B7C3E0(*a8, (__int64)v61, 0, (__int64)v63, v64);
      v63 = *(int **)(v41 + 8);
      v71 = 1;
      v61 = v80;
    }
    v72 = (unsigned int)v113;
    v73 = v112;
    if ( !(_DWORD)v113 )
      goto LABEL_99;
    v74 = 0;
    while ( 2 )
    {
      while ( 2 )
      {
        v75 = v74;
        v76 = &v73[v74];
        if ( v63[v74] != -1 )
        {
          if ( !v71 )
            v75 = v63[v74];
          *v76 = v72 + v75;
          v73 = v112;
LABEL_94:
          if ( (unsigned int)v72 == ++v74 )
            goto LABEL_98;
          continue;
        }
        break;
      }
      if ( *v76 == -1 )
        goto LABEL_94;
      *v76 = v74++;
      v73 = v112;
      if ( (unsigned int)v72 != v74 )
        continue;
      break;
    }
LABEL_98:
    v72 = (unsigned int)v113;
LABEL_99:
    if ( !v15 )
      v15 = *(__int64 ****)(*(_QWORD *)a10 - 96LL);
    v41 += 72;
    v15 = sub_2B7C3E0(*(__int64 **)(a10 + 8), (__int64)v15, (__int64)v61, (__int64)v73, v72);
    if ( v107 != v41 )
      continue;
    break;
  }
LABEL_102:
  v77 = v111;
  if ( (v111 & 1) == 0 && v111 )
  {
    if ( *(_QWORD *)v111 != v111 + 16 )
      _libc_free(*(_QWORD *)v111);
    j_j___libc_free_0(v77);
  }
  v78 = (unsigned __int64)v110;
  if ( ((unsigned __int8)v110 & 1) == 0 && v110 )
  {
    if ( (unsigned __int64 *)*v110 != v110 + 2 )
      _libc_free(*v110);
    j_j___libc_free_0(v78);
  }
  if ( v112 != (_DWORD *)v114 )
    _libc_free((unsigned __int64)v112);
  return v15;
}
