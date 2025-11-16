// Function: sub_8F2910
// Address: 0x8f2910
//
__int64 *__fastcall sub_8F2910(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // esi
  unsigned __int8 *v11; // r14
  _BOOL4 v12; // eax
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // r10
  __int64 v16; // r11
  __int64 v17; // r14
  __int64 v18; // rax
  int v19; // r8d
  unsigned __int64 v20; // rsi
  __int64 v21; // r9
  unsigned int *v22; // rdi
  unsigned int *v23; // r9
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  __int64 v26; // r8
  _DWORD *v27; // rax
  int v28; // ebx
  __int64 v29; // r13
  __int64 v30; // rdi
  __int64 v31; // r14
  __int64 v32; // r12
  unsigned int *v33; // r9
  unsigned int *v34; // r10
  __int64 v35; // r11
  int v36; // r8d
  unsigned int *v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rcx
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  int v42; // r8d
  unsigned int *v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rcx
  unsigned __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r12
  __int64 v49; // r14
  __int64 v50; // rsi
  __int64 v51; // r13
  __int64 v52; // r9
  int v53; // r10d
  unsigned int *v54; // r11
  int v55; // edi
  unsigned int *v56; // rdx
  unsigned __int64 v57; // rax
  __int64 v58; // rcx
  unsigned __int64 v59; // rax
  __int64 v60; // rdx
  int i; // ebx
  int v62; // edi
  unsigned int *v63; // rdx
  unsigned __int64 v64; // rax
  __int64 v65; // rcx
  unsigned __int64 v66; // rax
  __int64 v67; // rdx
  char v68; // al
  __int64 v69; // rdx
  int v70; // r10d
  char v71; // al
  int v72; // r8d
  __int64 v73; // r9
  int v74; // edx
  int v75; // edx
  int v76; // esi
  int v77; // r8d
  _QWORD *v78; // rbx
  __int64 v79; // r11
  int v80; // edx
  int v82; // esi
  unsigned int *v83; // rdx
  unsigned __int64 v84; // rax
  __int64 v85; // r8
  unsigned __int64 v86; // rax
  __int64 v87; // rdx
  int v88; // eax
  int v89; // r10d
  char v90; // r11
  __int64 v91; // rax
  int v92; // eax
  __int64 v93; // rax
  __int64 v94; // rax
  char v95; // dl
  int v96; // r8d
  __int64 v97; // r11
  __int64 v98; // rax
  char v99; // dl
  __int64 v100; // rax
  char v101; // r9
  int v102; // eax
  __int64 v103; // r10
  char v104; // al
  __int64 *v105; // [rsp+0h] [rbp-80h]
  void *dest; // [rsp+8h] [rbp-78h]
  int v107; // [rsp+10h] [rbp-70h]
  __int64 v108; // [rsp+10h] [rbp-70h]
  int v109; // [rsp+10h] [rbp-70h]
  __int64 v110; // [rsp+18h] [rbp-68h]
  _QWORD *v111; // [rsp+18h] [rbp-68h]
  _BOOL4 v113; // [rsp+24h] [rbp-5Ch]
  __int64 v115; // [rsp+30h] [rbp-50h]
  int v116; // [rsp+30h] [rbp-50h]
  __int64 v117; // [rsp+30h] [rbp-50h]
  char v119; // [rsp+38h] [rbp-48h]
  int v120; // [rsp+40h] [rbp-40h]
  int v121; // [rsp+40h] [rbp-40h]
  int src; // [rsp+48h] [rbp-38h]
  unsigned int *srca; // [rsp+48h] [rbp-38h]

  v5 = *(_QWORD *)qword_4F690E0;
  v110 = qword_4F690E0;
  *(_DWORD *)(qword_4F690E0 + 2088) = 0;
  v6 = *(_QWORD *)v5;
  *(_DWORD *)(v5 + 2088) = 0;
  *(_DWORD *)(v6 + 2088) = 0;
  v7 = *(_QWORD *)v6;
  *(_DWORD *)(v7 + 2088) = 0;
  v8 = **(_QWORD **)v7;
  v105 = *(__int64 **)v7;
  *(_DWORD *)(*(_QWORD *)v7 + 2088LL) = 0;
  qword_4F690E0 = v8;
  v9 = *(int *)(a2 + 8);
  v10 = *(_DWORD *)(a2 + 28);
  if ( (_DWORD)v9 == v10 )
  {
    v11 = (unsigned __int8 *)(a2 + 12);
    src = 30103 * v9 / 100000;
    v12 = sub_8EE3E0((unsigned __int8 *)(a2 + 12), v10);
    if ( !v73 )
    {
      src = 0;
      v107 = 0;
      *(_DWORD *)(v6 + 8) = 1;
      v113 = v12;
      *(_DWORD *)(v6 + 2088) = 1;
      goto LABEL_90;
    }
    v74 = 0;
    if ( (int)v73 < 0 )
    {
      v107 = v73;
      v75 = -(int)v73;
      v76 = -(int)v73;
      goto LABEL_92;
    }
LABEL_52:
    v14 = v72 + src;
    v107 = src;
    v13 = v74;
    goto LABEL_4;
  }
  v11 = (unsigned __int8 *)(a2 + 12);
  src = 30103 * v9 / 100000;
  v12 = sub_8EE3E0((unsigned __int8 *)(a2 + 12), v10);
  if ( !v15 )
  {
    v107 = 0;
    src = 0;
LABEL_4:
    if ( v13 != v14 )
    {
      if ( v13 >= v14 )
      {
        v120 = 1;
        *(_DWORD *)(v6 + 8) = 1;
        v14 = v13 - v14;
        *(_DWORD *)(v6 + 2088) = 1;
        v113 = v12;
LABEL_7:
        sub_8F0920(v6, v14);
        goto LABEL_8;
      }
      *(_DWORD *)(v6 + 8) = 1;
      *(_DWORD *)(v6 + 2088) = 1;
      v113 = v12;
      v120 = v14 - v13 + 1;
LABEL_8:
      sub_8EEB70(v110, v11, *(_DWORD *)(a2 + 28));
      v111 = (_QWORD *)v16;
      v17 = sub_8F0A50(v16, v6);
      sub_8F0920(v17, v113 + 1);
      *v111 = qword_4F690E0;
      qword_4F690E0 = (__int64)v111;
      *(_DWORD *)(v5 + 8) = 1;
      *(_DWORD *)(v5 + 2088) = 1;
      if ( src )
      {
        sub_8F0C00(v5, src);
        v111 = (_QWORD *)qword_4F690E0;
      }
      goto LABEL_10;
    }
    *(_DWORD *)(v6 + 8) = 1;
    *(_DWORD *)(v6 + 2088) = 1;
    v113 = v12;
    v120 = v14 + 1;
    if ( v14 )
      goto LABEL_7;
LABEL_90:
    v120 = 1;
    goto LABEL_8;
  }
  v74 = v13;
  v72 = v14;
  if ( (int)v15 >= 0 )
    goto LABEL_52;
  v75 = v74 - v15;
  v76 = -(int)v15;
  if ( v14 == v75 )
  {
    v107 = v15;
    *(_DWORD *)(v6 + 8) = 1;
    *(_DWORD *)(v6 + 2088) = 1;
    v113 = v12;
    v120 = v14 + 1;
LABEL_56:
    v116 = v14;
    sub_8F0C00(v6, v76);
    src = 0;
    v14 = v116;
    goto LABEL_7;
  }
  v107 = v15;
LABEL_92:
  if ( v75 >= v72 )
  {
    v120 = 1;
    *(_DWORD *)(v6 + 8) = 1;
    v14 = v75 - v72;
    *(_DWORD *)(v6 + 2088) = 1;
    v113 = v12;
    goto LABEL_56;
  }
  *(_DWORD *)(v6 + 8) = 1;
  *(_DWORD *)(v6 + 2088) = 1;
  v120 = v72 - v75 + 1;
  v113 = v12;
  sub_8F0C00(v6, v76);
  sub_8EEB70(v110, v11, *(_DWORD *)(a2 + 28));
  v111 = (_QWORD *)v97;
  v17 = sub_8F0A50(v97, v6);
  sub_8F0920(v17, v113 + 1);
  *v111 = qword_4F690E0;
  qword_4F690E0 = (__int64)v111;
  *(_DWORD *)(v5 + 8) = 1;
  *(_DWORD *)(v5 + 2088) = 1;
LABEL_10:
  sub_8F0920(v5, v113 + v120);
  v18 = *(unsigned int *)(v5 + 2088);
  *(_DWORD *)(v7 + 2088) = v18;
  dest = (void *)(v7 + 8);
  memcpy((void *)(v7 + 8), (const void *)(v5 + 8), 4 * v18);
  v19 = *(_DWORD *)(v7 + 2088);
  if ( v19 )
  {
    v20 = 0;
    v21 = v7 + 4LL * v19;
    v22 = (unsigned int *)(v21 + 4);
    v23 = (unsigned int *)(v21 - 4LL * (unsigned int)(v19 - 1));
    do
    {
      v24 = *v22--;
      v25 = v24 | v20;
      v22[1] = v25 / 0xA;
      v20 = (v25 % 0xA) << 32;
    }
    while ( v23 != v22 );
    if ( v20 )
    {
      sub_8EEC80(v7, 1u);
      v19 = *(_DWORD *)(v7 + 2088);
    }
    if ( v19 )
    {
      v26 = (unsigned int)(v19 - 1);
      v27 = (_DWORD *)(v7 + 4 * v26 + 8);
      while ( !*v27 )
      {
        *(_DWORD *)(v7 + 2088) = v26;
        --v27;
        if ( !(_DWORD)v26 )
          break;
        LODWORD(v26) = v26 - 1;
      }
    }
  }
  v115 = v5;
  v28 = v107;
  v108 = a1;
  v29 = v17;
  v30 = v17;
  srca = (unsigned int *)(v17 + 8);
  v31 = v6;
  v32 = v6 + 12;
  if ( (int)sub_8EECF0(v30, v7) < 0 )
  {
    do
    {
      v36 = *(_DWORD *)(v29 + 2088);
      --v28;
      if ( v36 > 0 )
      {
        v37 = v33;
        v38 = 0;
        do
        {
          v39 = *v37++;
          v40 = v38 + 10 * v39;
          *(v37 - 1) = v40;
          v38 = HIDWORD(v40);
        }
        while ( (unsigned int *)(v35 + 4LL * (unsigned int)(v36 - 1)) != v37 );
        if ( v38 )
        {
          v41 = v36++;
          v33[v41] = v38;
        }
      }
      *(_DWORD *)(v29 + 2088) = v36;
      v42 = *(_DWORD *)(v31 + 2088);
      if ( v42 > 0 )
      {
        v43 = v34;
        v44 = 0;
        do
        {
          v45 = *v43++;
          v46 = v44 + 10 * v45;
          *(v43 - 1) = v46;
          v44 = HIDWORD(v46);
        }
        while ( (unsigned int *)(v32 + 4LL * (unsigned int)(v42 - 1)) != v43 );
        if ( v44 )
        {
          v47 = v42++;
          v34[v47] = v44;
        }
      }
      *(_DWORD *)(v31 + 2088) = v42;
    }
    while ( (int)sub_8EECF0(v29, v7) < 0 );
  }
  v48 = v31;
  v49 = v29;
  v50 = v29;
  v51 = v108;
  if ( (int)sub_8EECF0(v115, v50) <= 0 )
  {
    do
    {
      v55 = *(_DWORD *)(v115 + 2088);
      if ( v55 > 0 )
      {
        v56 = v54;
        v57 = 0;
        do
        {
          v58 = *v56++;
          v59 = v57 + 10 * v58;
          *(v56 - 1) = v59;
          v57 = HIDWORD(v59);
        }
        while ( (unsigned int *)(v52 + 4LL * (unsigned int)(v55 - 1)) != v56 );
        if ( v57 )
        {
          v60 = v55++;
          v54[v60] = v57;
        }
      }
      *(_DWORD *)(v115 + 2088) = v55;
    }
    while ( (int)sub_8EECF0(v115, v49) <= 0 );
  }
  v109 = v53;
  if ( a4 )
    sub_8F0920(v48, a4);
  for ( i = 0; ; i = v70 )
  {
    v62 = *(_DWORD *)(v49 + 2088);
    if ( v62 > 0 )
    {
      v63 = srca;
      v64 = 0;
      do
      {
        v65 = *v63++;
        v66 = v64 + 10 * v65;
        *(v63 - 1) = v66;
        v64 = HIDWORD(v66);
      }
      while ( (unsigned int *)(v49 + 12 + 4LL * (unsigned int)(v62 - 1)) != v63 );
      if ( v64 )
      {
        v67 = v62++;
        srca[v67] = v64;
      }
    }
    *(_DWORD *)(v49 + 2088) = v62;
    v68 = sub_8EEDF0(v49, v115);
    v69 = i;
    v70 = i + 1;
    v71 = v68 + 48;
    *(_BYTE *)(v51 + i + 8) = v71;
    if ( a3 != 0x7FFFFFFF )
    {
      if ( a3 == i )
      {
        v77 = i;
        v78 = (_QWORD *)v115;
        v79 = v69;
        if ( v71 > 52 && (v71 != 53 || *(_DWORD *)(v49 + 2088) || (*(_BYTE *)(v51 + v77 - 1 + 8) & 1) != 0) )
        {
          if ( v77 )
          {
            v94 = v69;
            do
            {
              v95 = *(_BYTE *)(v51 + v94 + 7);
              v96 = v94;
              *(_BYTE *)(v51 + v94 + 7) = v95 + 1;
              if ( v95 != 57 )
                goto LABEL_87;
            }
            while ( (_DWORD)--v94 );
          }
LABEL_97:
          *(_BYTE *)(v51 + 8) = 49;
        }
        else if ( v77 )
        {
          goto LABEL_79;
        }
        ++v109;
        v80 = 1;
        goto LABEL_63;
      }
      continue;
    }
    v82 = *(_DWORD *)(v48 + 2088);
    if ( v82 > 0 )
    {
      v83 = (unsigned int *)(v48 + 8);
      v84 = 0;
      do
      {
        v85 = *v83++;
        v86 = v84 + 10 * v85;
        *(v83 - 1) = v86;
        v84 = HIDWORD(v86);
      }
      while ( (unsigned int *)(v48 + 4LL * (unsigned int)(v82 - 1) + 12) != v83 );
      if ( v84 )
      {
        v87 = v82++;
        *(_DWORD *)(v48 + 8 + 4 * v87) = v84;
      }
    }
    *(_DWORD *)(v48 + 2088) = v82;
    if ( !i )
      continue;
    v88 = sub_8EECF0(v49, v48);
    if ( v88 < 0 || !v88 && (*(_BYTE *)(a2 + 12) & 1) == 0 )
      break;
    v91 = *(unsigned int *)(v49 + 2088);
    *(_DWORD *)(v7 + 2088) = v91;
    memcpy(dest, srca, 4 * v91);
    sub_8EFAB0(v7, v48);
    if ( v113 )
      sub_8EFAB0(v7, v48);
    v92 = sub_8EECF0(v115, v7);
    if ( v92 < 0 || !v92 && (*(_BYTE *)(a2 + 12) & 1) == 0 )
    {
      v78 = (_QWORD *)v115;
      v79 = v70;
      if ( !*(_DWORD *)(v49 + 2088) )
        goto LABEL_79;
      v98 = v70;
      while ( 1 )
      {
        v99 = *(_BYTE *)(v51 + v98 + 7);
        v96 = v98;
        *(_BYTE *)(v51 + v98 + 7) = v99 + 1;
        if ( v99 != 57 )
          goto LABEL_87;
        if ( (int)--v98 <= 0 )
          goto LABEL_97;
      }
    }
  }
  v100 = *(unsigned int *)(v49 + 2088);
  v101 = v90;
  v78 = (_QWORD *)v115;
  v79 = v89;
  if ( (_DWORD)v100 )
  {
    v121 = v89;
    *(_DWORD *)(v7 + 2088) = v100;
    v117 = v89;
    v119 = v101;
    memcpy(dest, srca, 4 * v100);
    sub_8F0920(v7, 1);
    v102 = sub_8EECF0((__int64)v78, v7);
    v103 = v121;
    if ( v102 < 0 || (v79 = v117, !v102) && (v119 & 1) != 0 )
    {
      while ( 1 )
      {
        v104 = *(_BYTE *)(v51 + v103 + 7);
        v96 = v103;
        *(_BYTE *)(v51 + v103 + 7) = v104 + 1;
        if ( v104 != 57 )
          break;
        if ( (int)--v103 <= 0 )
          goto LABEL_97;
      }
LABEL_87:
      v79 = v96;
    }
  }
LABEL_79:
  v93 = v79;
  do
  {
    v80 = v93;
    if ( (int)v93 <= 1 )
      break;
    --v93;
  }
  while ( *(_BYTE *)(v51 + v93 + 8) == 48 );
LABEL_63:
  qword_4F690E0 = v49;
  *(_DWORD *)v51 = *(_DWORD *)(a2 + 4);
  *(_DWORD *)(v51 + 4) = v109;
  *(_BYTE *)(v51 + v80 + 8) = 0;
  *(_DWORD *)(v51 + 48) = v80;
  *v105 = (__int64)v111;
  *(_QWORD *)v7 = v105;
  *(_QWORD *)v48 = v7;
  *v78 = v48;
  *(_QWORD *)v49 = v78;
  return v105;
}
