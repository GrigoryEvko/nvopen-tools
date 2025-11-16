// Function: sub_1DDD700
// Address: 0x1ddd700
//
__int64 __fastcall sub_1DDD700(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rdx
  _QWORD *v10; // rdi
  unsigned int v11; // esi
  int v12; // eax
  _QWORD *v13; // rax
  __int64 v14; // r12
  __int64 v15; // r11
  int v16; // r15d
  __int64 v17; // r14
  __int64 v18; // r11
  unsigned int v19; // ecx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  int v22; // esi
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // ecx
  int v26; // edx
  __int64 v27; // rdi
  int v28; // r15d
  int v29; // ebx
  _BYTE *v30; // rdi
  size_t v31; // rdx
  char *v32; // rsi
  _BYTE *v33; // rax
  void **v34; // r8
  __int64 v35; // rax
  _DWORD *v36; // rdx
  _BYTE *v38; // rdi
  size_t v39; // rdx
  char *v40; // rsi
  unsigned __int64 v41; // rax
  void **v42; // r8
  unsigned int v43; // r8d
  __int64 *v44; // rax
  __int64 v45; // rcx
  _WORD *v46; // rdx
  __int64 v47; // rax
  int v48; // r9d
  _QWORD *v49; // r10
  int v50; // edx
  int v51; // ecx
  int v52; // ecx
  __int64 v53; // rdi
  _QWORD *v54; // r8
  unsigned int v55; // r13d
  int v56; // r10d
  __int64 v57; // rsi
  unsigned int v58; // edx
  unsigned int v59; // eax
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  __int64 v63; // r12
  int v64; // eax
  __int64 v65; // r12
  _QWORD *v66; // rax
  __int64 v67; // rdx
  _QWORD *i; // rdx
  __int64 v69; // rax
  int v70; // r10d
  __int64 *v71; // rdx
  int v72; // eax
  int v73; // eax
  int v74; // eax
  int v75; // esi
  __int64 v76; // rdi
  unsigned int v77; // ecx
  __int64 v78; // r8
  int v79; // r10d
  __int64 *v80; // r9
  int v81; // eax
  int v82; // ecx
  __int64 v83; // rdi
  int v84; // r9d
  unsigned int v85; // r12d
  __int64 *v86; // r8
  __int64 v87; // rsi
  int v88; // r13d
  _QWORD *v89; // r10
  _QWORD *v90; // rax
  _QWORD *v91; // r11
  __int64 v92; // [rsp+0h] [rbp-90h]
  __int64 v93; // [rsp+0h] [rbp-90h]
  __int64 v94; // [rsp+8h] [rbp-88h]
  int v95; // [rsp+8h] [rbp-88h]
  __int64 v96; // [rsp+10h] [rbp-80h]
  size_t v97; // [rsp+10h] [rbp-80h]
  size_t v98; // [rsp+10h] [rbp-80h]
  __int64 v100; // [rsp+20h] [rbp-70h] BYREF
  char v101; // [rsp+28h] [rbp-68h]
  void *v102; // [rsp+30h] [rbp-60h] BYREF
  __int64 v103; // [rsp+38h] [rbp-58h]
  _BYTE *v104; // [rsp+40h] [rbp-50h]
  void *dest; // [rsp+48h] [rbp-48h]
  int v106; // [rsp+50h] [rbp-40h]
  __int64 v107; // [rsp+58h] [rbp-38h]

  v4 = a3;
  v5 = a1;
  if ( *(_BYTE *)a2 )
  {
    v28 = -1;
    v29 = qword_4FC4640[20];
    if ( !v29 )
      goto LABEL_24;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    v106 = 1;
    dest = 0;
    v104 = 0;
    v103 = 0;
    v102 = &unk_49EFBE0;
    v107 = a1;
    goto LABEL_39;
  }
  v6 = *(_QWORD *)(a2 + 16);
  v7 = *(_QWORD *)(a3 + 56);
  v8 = a2 + 24;
  v10 = *(_QWORD **)(a2 + 32);
  v11 = *(_DWORD *)(a2 + 48);
  v96 = v8;
  if ( !v6 )
    goto LABEL_11;
  if ( v6 == v7 )
    goto LABEL_47;
  v12 = *(_DWORD *)(a2 + 40);
  ++*(_QWORD *)(a2 + 24);
  if ( !v12 )
  {
    if ( !*(_DWORD *)(a2 + 44) )
      goto LABEL_11;
    if ( v11 > 0x40 )
    {
      j___libc_free_0(v10);
      *(_DWORD *)(a2 + 48) = 0;
      v11 = 0;
      v10 = 0;
      *(_QWORD *)(a2 + 32) = 0;
      v15 = v7 + 320;
      *(_QWORD *)(a2 + 40) = 0;
      *(_QWORD *)(a2 + 16) = v7;
      v14 = *(_QWORD *)(v7 + 328);
      if ( v14 == v7 + 320 )
      {
LABEL_102:
        ++*(_QWORD *)(a2 + 24);
        v11 = 0;
        goto LABEL_103;
      }
      goto LABEL_12;
    }
    goto LABEL_7;
  }
  v58 = 4 * v12;
  if ( (unsigned int)(4 * v12) < 0x40 )
    v58 = 64;
  if ( v58 >= v11 )
  {
LABEL_7:
    v13 = &v10[2 * v11];
    if ( v13 != v10 )
    {
      do
      {
        *v10 = -8;
        v10 += 2;
      }
      while ( v13 != v10 );
      v10 = *(_QWORD **)(a2 + 32);
      v11 = *(_DWORD *)(a2 + 48);
    }
    *(_QWORD *)(a2 + 40) = 0;
    goto LABEL_11;
  }
  v59 = v12 - 1;
  if ( !v59 )
  {
    v65 = 2048;
    v64 = 128;
LABEL_78:
    v95 = v64;
    j___libc_free_0(v10);
    *(_DWORD *)(a2 + 48) = v95;
    v66 = (_QWORD *)sub_22077B0(v65);
    v67 = *(unsigned int *)(a2 + 48);
    *(_QWORD *)(a2 + 40) = 0;
    *(_QWORD *)(a2 + 32) = v66;
    v10 = v66;
    v11 = v67;
    for ( i = &v66[2 * v67]; i != v66; v66 += 2 )
    {
      if ( v66 )
        *v66 = -8;
    }
    goto LABEL_11;
  }
  _BitScanReverse(&v59, v59);
  v60 = (unsigned int)(1 << (33 - (v59 ^ 0x1F)));
  if ( (int)v60 < 64 )
    v60 = 64;
  if ( (_DWORD)v60 != v11 )
  {
    v61 = (4 * (int)v60 / 3u + 1) | ((unsigned __int64)(4 * (int)v60 / 3u + 1) >> 1);
    v62 = ((v61 | (v61 >> 2)) >> 4) | v61 | (v61 >> 2) | ((((v61 | (v61 >> 2)) >> 4) | v61 | (v61 >> 2)) >> 8);
    v63 = (v62 | (v62 >> 16)) + 1;
    v64 = (v62 | (v62 >> 16)) + 1;
    v65 = 16 * v63;
    goto LABEL_78;
  }
  *(_QWORD *)(a2 + 40) = 0;
  v90 = &v10[2 * v60];
  do
  {
    if ( v10 )
      *v10 = -8;
    v10 += 2;
  }
  while ( v90 != v10 );
  v10 = *(_QWORD **)(a2 + 32);
  v11 = *(_DWORD *)(a2 + 48);
LABEL_11:
  *(_QWORD *)(a2 + 16) = v7;
  v14 = *(_QWORD *)(v7 + 328);
  v15 = v7 + 320;
  if ( v14 == v7 + 320 )
    goto LABEL_47;
LABEL_12:
  v94 = v4;
  v16 = 0;
  v17 = v15;
  v18 = v5;
  while ( 1 )
  {
    if ( v11 )
    {
      v19 = (v11 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v20 = &v10[2 * v19];
      v21 = *v20;
      if ( v14 == *v20 )
        goto LABEL_14;
      v48 = 1;
      v49 = 0;
      while ( v21 != -8 )
      {
        if ( v21 == -16 && !v49 )
          v49 = v20;
        v19 = (v11 - 1) & (v48 + v19);
        v20 = &v10[2 * v19];
        v21 = *v20;
        if ( *v20 == v14 )
          goto LABEL_14;
        ++v48;
      }
      v50 = *(_DWORD *)(a2 + 40);
      if ( v49 )
        v20 = v49;
      ++*(_QWORD *)(a2 + 24);
      v26 = v50 + 1;
      if ( 4 * v26 < 3 * v11 )
      {
        if ( v11 - (v26 + *(_DWORD *)(a2 + 44)) <= v11 >> 3 )
        {
          v93 = v18;
          sub_1DDD540(v96, v11);
          v51 = *(_DWORD *)(a2 + 48);
          if ( !v51 )
            goto LABEL_153;
          v52 = v51 - 1;
          v53 = *(_QWORD *)(a2 + 32);
          v54 = 0;
          v55 = v52 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v18 = v93;
          v56 = 1;
          v26 = *(_DWORD *)(a2 + 40) + 1;
          v20 = (_QWORD *)(v53 + 16LL * v55);
          v57 = *v20;
          if ( *v20 != v14 )
          {
            while ( v57 != -8 )
            {
              if ( !v54 && v57 == -16 )
                v54 = v20;
              v55 = v52 & (v56 + v55);
              v20 = (_QWORD *)(v53 + 16LL * v55);
              v57 = *v20;
              if ( *v20 == v14 )
                goto LABEL_20;
              ++v56;
            }
            if ( v54 )
              v20 = v54;
          }
        }
        goto LABEL_20;
      }
    }
    else
    {
      ++*(_QWORD *)(a2 + 24);
    }
    v92 = v18;
    sub_1DDD540(v96, 2 * v11);
    v22 = *(_DWORD *)(a2 + 48);
    if ( !v22 )
      goto LABEL_153;
    v23 = v22 - 1;
    v24 = *(_QWORD *)(a2 + 32);
    v18 = v92;
    v25 = v23 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v26 = *(_DWORD *)(a2 + 40) + 1;
    v20 = (_QWORD *)(v24 + 16LL * v25);
    v27 = *v20;
    if ( v14 != *v20 )
    {
      v88 = 1;
      v89 = 0;
      while ( v27 != -8 )
      {
        if ( !v89 && v27 == -16 )
          v89 = v20;
        v25 = v23 & (v88 + v25);
        v20 = (_QWORD *)(v24 + 16LL * v25);
        v27 = *v20;
        if ( *v20 == v14 )
          goto LABEL_20;
        ++v88;
      }
      if ( v89 )
        v20 = v89;
    }
LABEL_20:
    *(_DWORD *)(a2 + 40) = v26;
    if ( *v20 != -8 )
      --*(_DWORD *)(a2 + 44);
    *v20 = v14;
    *((_DWORD *)v20 + 2) = 0;
LABEL_14:
    *((_DWORD *)v20 + 2) = v16;
    v14 = *(_QWORD *)(v14 + 8);
    ++v16;
    if ( v14 == v17 )
      break;
    v10 = *(_QWORD **)(a2 + 32);
    v11 = *(_DWORD *)(a2 + 48);
  }
  v4 = v94;
  v10 = *(_QWORD **)(a2 + 32);
  v5 = v18;
  v11 = *(_DWORD *)(a2 + 48);
LABEL_47:
  if ( !v11 )
    goto LABEL_102;
  v43 = (v11 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v44 = &v10[2 * v43];
  v45 = *v44;
  if ( *v44 == v4 )
  {
    v28 = *((_DWORD *)v44 + 2);
    goto LABEL_50;
  }
  v70 = 1;
  v71 = 0;
  while ( 2 )
  {
    if ( v45 == -8 )
    {
      if ( !v71 )
        v71 = v44;
      v72 = *(_DWORD *)(a2 + 40);
      ++*(_QWORD *)(a2 + 24);
      v73 = v72 + 1;
      if ( 4 * v73 < 3 * v11 )
      {
        if ( v11 - (v73 + *(_DWORD *)(a2 + 44)) > v11 >> 3 )
        {
LABEL_97:
          *(_DWORD *)(a2 + 40) = v73;
          if ( *v71 != -8 )
            --*(_DWORD *)(a2 + 44);
          *v71 = v4;
          v28 = 0;
          *((_DWORD *)v71 + 2) = 0;
          v29 = qword_4FC4640[20];
          if ( !v29 )
            goto LABEL_24;
          *(_BYTE *)(v5 + 16) = 0;
          v28 = 0;
          *(_QWORD *)v5 = v5 + 16;
          *(_QWORD *)(v5 + 8) = 0;
          v106 = 1;
          dest = 0;
          v104 = 0;
          v103 = 0;
          v102 = &unk_49EFBE0;
          v107 = v5;
          goto LABEL_26;
        }
        sub_1DDD540(v96, v11);
        v81 = *(_DWORD *)(a2 + 48);
        if ( v81 )
        {
          v82 = v81 - 1;
          v83 = *(_QWORD *)(a2 + 32);
          v84 = 1;
          v85 = (v81 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v86 = 0;
          v73 = *(_DWORD *)(a2 + 40) + 1;
          v71 = (__int64 *)(v83 + 16LL * v85);
          v87 = *v71;
          if ( *v71 != v4 )
          {
            while ( v87 != -8 )
            {
              if ( v87 == -16 && !v86 )
                v86 = v71;
              v85 = v82 & (v84 + v85);
              v71 = (__int64 *)(v83 + 16LL * v85);
              v87 = *v71;
              if ( *v71 == v4 )
                goto LABEL_97;
              ++v84;
            }
            if ( v86 )
              v71 = v86;
          }
          goto LABEL_97;
        }
LABEL_153:
        ++*(_DWORD *)(a2 + 40);
        BUG();
      }
LABEL_103:
      sub_1DDD540(v96, 2 * v11);
      v74 = *(_DWORD *)(a2 + 48);
      if ( v74 )
      {
        v75 = v74 - 1;
        v76 = *(_QWORD *)(a2 + 32);
        v77 = (v74 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v73 = *(_DWORD *)(a2 + 40) + 1;
        v71 = (__int64 *)(v76 + 16LL * v77);
        v78 = *v71;
        if ( *v71 != v4 )
        {
          v79 = 1;
          v80 = 0;
          while ( v78 != -8 )
          {
            if ( !v80 && v78 == -16 )
              v80 = v71;
            v77 = v75 & (v79 + v77);
            v71 = (__int64 *)(v76 + 16LL * v77);
            v78 = *v71;
            if ( *v71 == v4 )
              goto LABEL_97;
            ++v79;
          }
          if ( v80 )
            v71 = v80;
        }
        goto LABEL_97;
      }
      goto LABEL_153;
    }
    if ( v45 != -16 || v71 )
      v44 = v71;
    v43 = (v11 - 1) & (v70 + v43);
    v91 = &v10[2 * v43];
    v45 = *v91;
    if ( *v91 != v4 )
    {
      ++v70;
      v71 = v44;
      v44 = &v10[2 * v43];
      continue;
    }
    break;
  }
  v28 = *((_DWORD *)v91 + 2);
LABEL_50:
  v29 = qword_4FC4640[20];
  if ( !v29 )
LABEL_24:
    v29 = dword_4FC4940;
  *(_QWORD *)(v5 + 8) = 0;
  *(_QWORD *)v5 = v5 + 16;
  *(_BYTE *)(v5 + 16) = 0;
  v106 = 1;
  dest = 0;
  v104 = 0;
  v103 = 0;
  v102 = &unk_49EFBE0;
  v107 = v5;
  if ( v28 == -1 )
  {
LABEL_39:
    v38 = dest;
    v40 = (char *)sub_1DD6290(v4);
    v41 = v104 - (_BYTE *)dest;
    if ( v104 - (_BYTE *)dest < v39 )
    {
      v69 = sub_16E7EE0((__int64)&v102, v40, v39);
      v38 = *(_BYTE **)(v69 + 24);
      v42 = (void **)v69;
      if ( *(_QWORD *)(v69 + 16) - (_QWORD)v38 > 2u )
      {
LABEL_43:
        v38[2] = 32;
        *(_WORD *)v38 = 14880;
        v42[3] = (char *)v42[3] + 3;
        goto LABEL_33;
      }
    }
    else
    {
      v42 = &v102;
      if ( v39 )
      {
        v97 = v39;
        memcpy(dest, v40, v39);
        v42 = &v102;
        dest = (char *)dest + v97;
        v38 = dest;
        v41 = v104 - (_BYTE *)dest;
      }
      if ( v41 > 2 )
        goto LABEL_43;
    }
    sub_16E7EE0((__int64)v42, " : ", 3u);
    goto LABEL_33;
  }
LABEL_26:
  v30 = dest;
  v32 = (char *)sub_1DD6290(v4);
  v33 = v104;
  if ( v104 - (_BYTE *)dest < v31 )
  {
    v34 = (void **)sub_16E7EE0((__int64)&v102, v32, v31);
    v30 = v34[3];
    if ( v30 != v34[2] )
      goto LABEL_30;
    goto LABEL_45;
  }
  v34 = &v102;
  if ( v31 )
  {
    v98 = v31;
    memcpy(dest, v32, v31);
    dest = (char *)dest + v98;
    v33 = v104;
    v30 = dest;
    v34 = &v102;
  }
  if ( v30 == v33 )
  {
LABEL_45:
    v34 = (void **)sub_16E7EE0((__int64)v34, "[", 1u);
    goto LABEL_31;
  }
LABEL_30:
  *v30 = 91;
  v34[3] = (char *)v34[3] + 1;
LABEL_31:
  v35 = sub_16E7AB0((__int64)v34, v28);
  v36 = *(_DWORD **)(v35 + 24);
  if ( *(_QWORD *)(v35 + 16) - (_QWORD)v36 <= 3u )
  {
    sub_16E7EE0(v35, "] : ", 4u);
  }
  else
  {
    *v36 = 540680285;
    *(_QWORD *)(v35 + 24) += 4LL;
  }
LABEL_33:
  switch ( v29 )
  {
    case 2:
      v47 = sub_1DDC3C0(a4, v4);
      sub_16E7A90((__int64)&v102, v47);
      break;
    case 3:
      sub_1DDC460((__int64)&v100, a4, v4);
      if ( v101 )
      {
        sub_16E7A90((__int64)&v102, v100);
      }
      else
      {
        v46 = dest;
        if ( (unsigned __int64)(v104 - (_BYTE *)dest) <= 6 )
        {
          sub_16E7EE0((__int64)&v102, "Unknown", 7u);
        }
        else
        {
          *(_DWORD *)dest = 1852534357;
          v46[2] = 30575;
          *((_BYTE *)v46 + 6) = 110;
          dest = (char *)dest + 7;
        }
      }
      break;
    case 1:
      sub_1DDC550(a4, (__int64)&v102, v4);
      break;
  }
  sub_16E7BC0((__int64 *)&v102);
  return v5;
}
