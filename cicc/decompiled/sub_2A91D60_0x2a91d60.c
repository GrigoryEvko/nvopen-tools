// Function: sub_2A91D60
// Address: 0x2a91d60
//
__int64 __fastcall sub_2A91D60(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v8; // rax
  _DWORD *v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // r13
  _BYTE *v12; // rbx
  _BYTE *v13; // rsi
  __int64 v14; // r8
  char v15; // al
  unsigned __int8 *v16; // rcx
  char v17; // al
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 *v21; // r13
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 *v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rsi
  unsigned __int64 v27; // rdx
  int v28; // esi
  unsigned __int64 *v29; // rdx
  unsigned __int64 v30; // r10
  int v31; // r13d
  unsigned __int64 v32; // rdi
  unsigned int i; // eax
  unsigned __int64 v34; // rcx
  bool v35; // bl
  unsigned __int64 v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rdx
  unsigned __int64 *v39; // rcx
  unsigned __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rcx
  unsigned __int64 *j; // r12
  __int64 v45; // rdi
  __int64 v46; // rsi
  unsigned __int64 v47; // rdx
  unsigned int v48; // ecx
  __int64 v49; // rax
  unsigned __int64 v50; // r13
  unsigned __int64 k; // rbx
  __int64 v52; // r14
  unsigned __int64 v53; // r15
  unsigned __int64 v54; // rdi
  __int64 v55; // r15
  unsigned __int64 v56; // r13
  unsigned __int64 m; // rbx
  __int64 v58; // r14
  unsigned __int64 v59; // r15
  unsigned __int64 v60; // rdi
  __int64 *v61; // rbx
  __int64 *v62; // r12
  __int64 v63; // rsi
  __int64 v64; // rdi
  __int64 *v65; // r12
  __int64 *v66; // rax
  __int64 *v67; // rbx
  __int64 *v68; // r13
  __int64 v69; // rdi
  unsigned int v70; // ecx
  __int64 v71; // rax
  __int64 *v72; // rbx
  __int64 v73; // rsi
  __int64 v74; // rdi
  unsigned __int64 v75; // r15
  __int64 v76; // rax
  unsigned __int64 *v77; // r12
  unsigned __int64 *v78; // r13
  unsigned __int64 v79; // rbx
  unsigned __int64 v80; // r14
  unsigned __int64 v81; // rdi
  unsigned int v82; // eax
  __int64 v83; // r13
  __int64 v84; // rax
  int v85; // ebx
  unsigned __int64 v86; // rdi
  int v87; // eax
  unsigned int v88; // [rsp+3Ch] [rbp-124h]
  _QWORD *v90; // [rsp+48h] [rbp-118h]
  _BYTE **v91; // [rsp+50h] [rbp-110h]
  _QWORD *v92; // [rsp+50h] [rbp-110h]
  __int64 *v93; // [rsp+50h] [rbp-110h]
  char *v94; // [rsp+50h] [rbp-110h]
  _QWORD *v95; // [rsp+58h] [rbp-108h]
  __int64 *v96; // [rsp+58h] [rbp-108h]
  unsigned __int64 *v97; // [rsp+68h] [rbp-F8h] BYREF
  char *v98; // [rsp+70h] [rbp-F0h] BYREF
  unsigned __int64 *v99; // [rsp+78h] [rbp-E8h]
  unsigned __int64 *v100; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v101; // [rsp+88h] [rbp-D8h]
  unsigned __int64 *v102; // [rsp+90h] [rbp-D0h] BYREF
  unsigned __int64 *v103; // [rsp+98h] [rbp-C8h]
  char *v104; // [rsp+A0h] [rbp-C0h]
  __int64 v105; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v106; // [rsp+B8h] [rbp-A8h]
  __int64 v107; // [rsp+C0h] [rbp-A0h]
  __int64 v108; // [rsp+C8h] [rbp-98h]
  _QWORD *v109; // [rsp+D0h] [rbp-90h] BYREF
  unsigned __int64 v110; // [rsp+D8h] [rbp-88h]
  _BYTE *v111; // [rsp+E0h] [rbp-80h]
  __int64 v112; // [rsp+E8h] [rbp-78h]
  _BYTE v113[32]; // [rsp+F0h] [rbp-70h] BYREF
  __int64 *v114; // [rsp+110h] [rbp-50h]
  __int64 v115; // [rsp+118h] [rbp-48h]
  _QWORD v116[8]; // [rsp+120h] [rbp-40h] BYREF

  if ( !a4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    *(_OWORD *)a1 = 0;
    return a1;
  }
  v8 = *(_QWORD *)(*(_QWORD *)(*a3 - 32LL) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  v9 = sub_AE2980(*(_QWORD *)(a2 + 48), *(_DWORD *)(v8 + 8) >> 8);
  v109 = 0;
  LODWORD(v9) = v9[3];
  v10 = (unsigned __int64)&v98;
  v110 = 0;
  v88 = (unsigned int)v9;
  v111 = v113;
  v112 = 0x400000000LL;
  v114 = v116;
  v115 = 0;
  v98 = (char *)&v98 + 4;
  v116[0] = 0;
  v116[1] = 0;
  v99 = (unsigned __int64 *)&v98;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v90 = &a3[a4];
  if ( v90 == a3 )
  {
    v102 = 0;
    v103 = 0;
    v104 = 0;
    goto LABEL_66;
  }
  v91 = (_BYTE **)a3;
  while ( 2 )
  {
    v11 = 64;
    v12 = *v91;
    while ( (char **)v10 != &v98 )
    {
      if ( sub_B445A0(*(_QWORD *)(v10 + 16), (__int64)v12) )
      {
        v13 = *(_BYTE **)(v10 + 16);
        v14 = (__int64)v12;
      }
      else
      {
        v14 = *(_QWORD *)(v10 + 16);
        v13 = (_BYTE *)v14;
      }
      v15 = *v12;
      v16 = 0;
      if ( *v12 > 0x1Cu && (v15 == 61 || v15 == 62) )
        v16 = (unsigned __int8 *)*((_QWORD *)v12 - 4);
      v17 = *v13;
      v18 = 0;
      if ( *v13 > 0x1Cu && (v17 == 61 || v17 == 62) )
        v18 = *((_QWORD *)v13 - 4);
      sub_2A906F0((__int64)&v102, a2, v18, v16, v14, 0);
      if ( (_BYTE)v104 )
      {
        v37 = *(_DWORD *)(v10 + 32);
        if ( v37 >= *(_DWORD *)(v10 + 36) )
        {
          v83 = sub_C8D7D0(v10 + 24, v10 + 40, 0, 0x18u, (unsigned __int64 *)&v97, v20);
          v101 = (unsigned int)v103;
          if ( (unsigned int)v103 > 0x40 )
            sub_C43780((__int64)&v100, (const void **)&v102);
          else
            v100 = v102;
          v84 = v83 + 24LL * *(unsigned int *)(v10 + 32);
          if ( v84 )
          {
            *(_QWORD *)v84 = v12;
            *(_DWORD *)(v84 + 16) = v101;
            *(_QWORD *)(v84 + 8) = v100;
            v101 = 0;
          }
          sub_969240((__int64 *)&v100);
          sub_2A8AC40((__int64 *)(v10 + 24), v83);
          v85 = (int)v97;
          v86 = *(_QWORD *)(v10 + 24);
          if ( v10 + 40 != v86 )
            _libc_free(v86);
          ++*(_DWORD *)(v10 + 32);
          *(_QWORD *)(v10 + 24) = v83;
          *(_DWORD *)(v10 + 36) = v85;
        }
        else
        {
          v101 = (unsigned int)v103;
          if ( (unsigned int)v103 > 0x40 )
          {
            sub_C43780((__int64)&v100, (const void **)&v102);
            v37 = *(_DWORD *)(v10 + 32);
            v38 = *(_QWORD *)(v10 + 24) + 24LL * v37;
            if ( v38 )
            {
LABEL_46:
              *(_QWORD *)v38 = v12;
              *(_DWORD *)(v38 + 16) = v101;
              *(_QWORD *)(v38 + 8) = v100;
              v37 = *(_DWORD *)(v10 + 32);
            }
            else if ( v101 > 0x40 && v100 )
            {
              j_j___libc_free_0_0((unsigned __int64)v100);
              v37 = *(_DWORD *)(v10 + 32);
            }
          }
          else
          {
            v100 = v102;
            v38 = *(_QWORD *)(v10 + 24) + 24LL * v37;
            if ( v38 )
              goto LABEL_46;
          }
          *(_DWORD *)(v10 + 32) = v37 + 1;
        }
        v39 = *(unsigned __int64 **)(v10 + 8);
        v40 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
        *v39 = v40 | *v39 & 7;
        *(_QWORD *)(v40 + 8) = v39;
        v41 = *(_QWORD *)v10;
        *(_QWORD *)(v10 + 8) = 0;
        v42 = v41 & 7;
        *(_QWORD *)v10 = v42;
        v29 = v99;
        v43 = *v99;
        *(_QWORD *)(v10 + 8) = v99;
        v34 = v43 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v10 = v34 | v42;
        *(_QWORD *)(v34 + 8) = v10;
        *v29 = *v29 & 7 | v10;
        if ( (_BYTE)v104 )
        {
          LOBYTE(v104) = 0;
          if ( (unsigned int)v103 > 0x40 )
          {
            v36 = (unsigned __int64)v102;
            if ( v102 )
              goto LABEL_40;
          }
        }
        goto LABEL_41;
      }
      v10 = *(_QWORD *)(v10 + 8);
      if ( !--v11 )
        break;
    }
    v101 = v88;
    if ( v88 > 0x40 )
      sub_C43690((__int64)&v100, 0, 0);
    else
      v100 = 0;
    v116[0] += 64LL;
    v21 = (unsigned __int64 *)(((unsigned __int64)v109 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v110 >= (unsigned __int64)(v21 + 8) && v109 )
      v109 = v21 + 8;
    else
      v21 = (unsigned __int64 *)sub_9D1E70((__int64)&v109, 64, 64, 3);
    v22 = (__int64)(v21 + 5);
    v21[2] = (unsigned __int64)v12;
    *v21 = 0;
    v21[1] = 0;
    v21[3] = (unsigned __int64)(v21 + 5);
    v21[4] = 0x100000000LL;
    v97 = v21;
    LODWORD(v103) = v101;
    if ( v101 <= 0x40 )
    {
      v102 = v100;
LABEL_30:
      *(_QWORD *)v22 = v12;
      *(_DWORD *)(v22 + 16) = (_DWORD)v103;
      *(_QWORD *)(v22 + 8) = v102;
      v23 = *((_DWORD *)v21 + 8);
      goto LABEL_31;
    }
    sub_C43780((__int64)&v102, (const void **)&v100);
    v23 = *((_DWORD *)v21 + 8);
    v22 = v21[3] + 24LL * v23;
    if ( v22 )
      goto LABEL_30;
    if ( (unsigned int)v103 > 0x40 && v102 )
    {
      j_j___libc_free_0_0((unsigned __int64)v102);
      v23 = *((_DWORD *)v21 + 8);
    }
LABEL_31:
    v24 = (__int64 *)v99;
    *((_DWORD *)v21 + 8) = v23 + 1;
    v25 = (__int64)v97;
    v26 = *v24;
    v27 = *v97;
    v97[1] = (unsigned __int64)v24;
    v26 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v25 = v26 | v27 & 7;
    *(_QWORD *)(v26 + 8) = v25;
    *v24 = *v24 & 7 | v25;
    v28 = v108;
    if ( !(_DWORD)v108 )
    {
      ++v105;
      v102 = 0;
      goto LABEL_142;
    }
    v29 = v97;
    v20 = (unsigned int)(v108 - 1);
    v30 = 0;
    v31 = 1;
    v32 = v97[2];
    for ( i = v20 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4)); ; i = v20 & v82 )
    {
      v34 = v106 + 8LL * i;
      v19 = *(_QWORD *)v34;
      if ( v97 == (unsigned __int64 *)-4096LL )
      {
        if ( v19 == -4096 )
          goto LABEL_38;
        v35 = v19 == -8192;
LABEL_128:
        if ( !v30 && v35 )
          v30 = v106 + 8LL * i;
        goto LABEL_131;
      }
      if ( v19 == -4096 )
        break;
      v35 = v19 == -8192;
      if ( v97 == (unsigned __int64 *)-8192LL )
      {
        if ( v19 == -8192 )
          goto LABEL_38;
        goto LABEL_128;
      }
      if ( v19 == -8192 )
        goto LABEL_128;
      if ( v32 == *(_QWORD *)(v19 + 16) )
        goto LABEL_38;
LABEL_131:
      v82 = v31 + i;
      ++v31;
    }
    if ( v30 )
      v34 = v30;
    ++v105;
    v87 = v107 + 1;
    v102 = (unsigned __int64 *)v34;
    if ( 4 * ((int)v107 + 1) >= (unsigned int)(3 * v108) )
    {
LABEL_142:
      v28 = 2 * v108;
      goto LABEL_143;
    }
    v19 = (unsigned int)v108 >> 3;
    if ( (int)v108 - HIDWORD(v107) - v87 <= (unsigned int)v19 )
    {
LABEL_143:
      sub_2A8BBE0((__int64)&v105, v28);
      sub_2A8A7B0((__int64)&v105, (__int64 *)&v97, &v102);
      v29 = v97;
      v34 = (unsigned __int64)v102;
      v87 = v107 + 1;
    }
    LODWORD(v107) = v87;
    if ( *(_QWORD *)v34 != -4096 )
      --HIDWORD(v107);
    *(_QWORD *)v34 = v29;
LABEL_38:
    if ( v101 > 0x40 )
    {
      v36 = (unsigned __int64)v100;
      if ( v100 )
LABEL_40:
        j_j___libc_free_0_0(v36);
    }
LABEL_41:
    if ( v90 != ++v91 )
    {
      v10 = (unsigned __int64)v99;
      continue;
    }
    break;
  }
  v102 = 0;
  v103 = 0;
  v104 = 0;
  if ( (_DWORD)v107 )
  {
    v75 = 40LL * (unsigned int)v107;
    v76 = sub_22077B0(v75);
    v77 = v103;
    v78 = v102;
    v94 = (char *)v76;
    if ( v103 != v102 )
    {
      do
      {
        v79 = *v78;
        v80 = *v78 + 24LL * *((unsigned int *)v78 + 2);
        if ( *v78 != v80 )
        {
          do
          {
            v80 -= 24LL;
            if ( *(_DWORD *)(v80 + 16) > 0x40u )
            {
              v81 = *(_QWORD *)(v80 + 8);
              if ( v81 )
                j_j___libc_free_0_0(v81);
            }
          }
          while ( v79 != v80 );
          v80 = *v78;
        }
        if ( (unsigned __int64 *)v80 != v78 + 2 )
          _libc_free(v80);
        v78 += 5;
      }
      while ( v77 != v78 );
      v78 = v102;
    }
    if ( v78 )
      j_j___libc_free_0((unsigned __int64)v78);
    v102 = (unsigned __int64 *)v94;
    v103 = (unsigned __int64 *)v94;
    v104 = &v94[v75];
  }
  for ( j = v99; j != (unsigned __int64 *)&v98; j = (unsigned __int64 *)j[1] )
  {
    if ( *((_DWORD *)j + 8) > 1u )
      sub_2A8C840((unsigned __int64 *)&v102, (__int64)(j + 3), (__int64)v29, v34, v19, v20);
  }
LABEL_66:
  v45 = v106;
  *(_QWORD *)a1 = v102;
  *(_QWORD *)(a1 + 8) = v103;
  *(_QWORD *)(a1 + 16) = v104;
  sub_C7D6A0(v45, 8LL * (unsigned int)v108, 8);
  v46 = (unsigned int)v112;
  v95 = v111;
  v92 = &v111[8 * (unsigned int)v112];
  if ( v111 != (_BYTE *)v92 )
  {
    v47 = (unsigned __int64)v111;
    while ( 1 )
    {
      v48 = (unsigned int)((__int64)((__int64)v95 - v47) >> 3) >> 7;
      v49 = 4096LL << v48;
      if ( v48 >= 0x1E )
        v49 = 0x40000000000LL;
      v50 = *v95 + v49;
      if ( *v95 == *(_QWORD *)(v47 + 8 * v46 - 8) )
        v50 = (unsigned __int64)v109;
      for ( k = ((*v95 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 64; k <= v50; k += 64LL )
      {
        v52 = *(_QWORD *)(k - 40);
        v53 = v52 + 24LL * *(unsigned int *)(k - 32);
        if ( v52 != v53 )
        {
          do
          {
            v53 -= 24LL;
            if ( *(_DWORD *)(v53 + 16) > 0x40u )
            {
              v54 = *(_QWORD *)(v53 + 8);
              if ( v54 )
                j_j___libc_free_0_0(v54);
            }
          }
          while ( v52 != v53 );
          v53 = *(_QWORD *)(k - 64 + 24);
        }
        if ( v53 != k - 24 )
          _libc_free(v53);
      }
      if ( v92 == ++v95 )
        break;
      v47 = (unsigned __int64)v111;
      v46 = (unsigned int)v112;
    }
  }
  v55 = 2LL * (unsigned int)v115;
  v96 = v114;
  v93 = &v114[v55];
  if ( v114 != &v114[v55] )
  {
    do
    {
      v56 = *v96 + v96[1];
      for ( m = ((*v96 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 64; v56 >= m; m += 64LL )
      {
        v58 = *(_QWORD *)(m - 40);
        v59 = v58 + 24LL * *(unsigned int *)(m - 32);
        if ( v58 != v59 )
        {
          do
          {
            v59 -= 24LL;
            if ( *(_DWORD *)(v59 + 16) > 0x40u )
            {
              v60 = *(_QWORD *)(v59 + 8);
              if ( v60 )
                j_j___libc_free_0_0(v60);
            }
          }
          while ( v58 != v59 );
          v59 = *(_QWORD *)(m - 64 + 24);
        }
        if ( v59 != m - 24 )
          _libc_free(v59);
      }
      v96 += 2;
    }
    while ( v93 != v96 );
    v61 = v114;
    v62 = &v114[2 * (unsigned int)v115];
    if ( v114 != v62 )
    {
      do
      {
        v63 = v61[1];
        v64 = *v61;
        v61 += 2;
        sub_C7D6A0(v64, v63, 16);
      }
      while ( v62 != v61 );
    }
  }
  LODWORD(v115) = 0;
  if ( (_DWORD)v112 )
  {
    v66 = (__int64 *)v111;
    v116[0] = 0;
    v67 = (__int64 *)&v111[8 * (unsigned int)v112];
    v68 = (__int64 *)(v111 + 8);
    v109 = *(_QWORD **)v111;
    v110 = (unsigned __int64)(v109 + 512);
    if ( v67 != (__int64 *)(v111 + 8) )
    {
      do
      {
        v69 = *v68;
        v70 = (unsigned int)(v68 - v66) >> 7;
        v71 = 4096LL << v70;
        if ( v70 >= 0x1E )
          v71 = 0x40000000000LL;
        ++v68;
        sub_C7D6A0(v69, v71, 16);
        v66 = (__int64 *)v111;
      }
      while ( v67 != v68 );
    }
    LODWORD(v112) = 1;
    sub_C7D6A0(*v66, 4096, 16);
    v72 = v114;
    v65 = &v114[2 * (unsigned int)v115];
    if ( v114 != v65 )
    {
      do
      {
        v73 = v72[1];
        v74 = *v72;
        v72 += 2;
        sub_C7D6A0(v74, v73, 16);
      }
      while ( v65 != v72 );
      goto LABEL_99;
    }
  }
  else
  {
LABEL_99:
    v65 = v114;
  }
  if ( v65 != v116 )
    _libc_free((unsigned __int64)v65);
  if ( v111 != v113 )
    _libc_free((unsigned __int64)v111);
  return a1;
}
