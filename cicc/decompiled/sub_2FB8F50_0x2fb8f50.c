// Function: sub_2FB8F50
// Address: 0x2fb8f50
//
void __fastcall sub_2FB8F50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  _BYTE *v13; // rax
  _BYTE *v14; // rcx
  _BYTE *i; // rdx
  __int64 *v16; // r13
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // r15
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdi
  _QWORD *v27; // rax
  int v28; // edx
  __int64 v29; // r8
  __int64 v30; // rdi
  int v31; // edx
  unsigned int v32; // eax
  int v33; // ecx
  _BYTE *v34; // r8
  _BYTE *v35; // rax
  char v36; // r10
  __int64 *v37; // r9
  __int64 v38; // rax
  __int64 *v39; // r12
  __int64 *m; // r14
  _BYTE *v41; // rax
  __int64 *v42; // r13
  __int64 *v43; // rbx
  _BYTE *v44; // rax
  __int64 v45; // rsi
  char *v46; // rdx
  char *v47; // rcx
  _QWORD *v48; // rax
  _QWORD *v49; // rsi
  __int64 v50; // r8
  __int64 v51; // r9
  char *v52; // rbx
  char *v53; // r12
  unsigned __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r13
  __int64 v57; // rdx
  char *v58; // rax
  bool v59; // zf
  unsigned int v60; // eax
  _BYTE *v61; // rbx
  unsigned __int64 v62; // r12
  __int64 v63; // rsi
  _BYTE *v64; // rax
  __int64 v65; // rcx
  __int64 v66; // r9
  unsigned __int64 v67; // r10
  __int64 v68; // rax
  __int64 v69; // r15
  unsigned __int64 v70; // r11
  __int64 v71; // rax
  __int64 v72; // r8
  char v73; // al
  __int64 v74; // rdx
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  _QWORD *v77; // rax
  __int64 v78; // rdx
  _QWORD *v79; // r10
  __int64 *v80; // rdi
  __int64 v81; // rdx
  _QWORD *v82; // r11
  unsigned int v83; // r8d
  __int64 v84; // r10
  __int64 *v85; // rdi
  char *v86; // rdx
  char *v87; // rax
  int k; // esi
  __int64 v90; // [rsp+18h] [rbp-3D8h]
  int v91; // [rsp+20h] [rbp-3D0h]
  unsigned int j; // [rsp+24h] [rbp-3CCh]
  __int64 v94; // [rsp+30h] [rbp-3C0h]
  __int64 v95; // [rsp+38h] [rbp-3B8h]
  __int64 v96; // [rsp+40h] [rbp-3B0h]
  int *v97; // [rsp+48h] [rbp-3A8h]
  __int64 v98; // [rsp+50h] [rbp-3A0h] BYREF
  void *s; // [rsp+58h] [rbp-398h]
  _BYTE v100[12]; // [rsp+60h] [rbp-390h]
  char v101; // [rsp+6Ch] [rbp-384h]
  char v102; // [rsp+70h] [rbp-380h] BYREF
  _BYTE *v103; // [rsp+B0h] [rbp-340h] BYREF
  __int64 v104; // [rsp+B8h] [rbp-338h]
  _BYTE v105[816]; // [rsp+C0h] [rbp-330h] BYREF

  v8 = sub_2DF8570(
         *(_QWORD *)(a1 + 8),
         *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL) + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL)),
         *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL),
         *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL),
         a5,
         a6);
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v104 = 0x800000000LL;
  v12 = *(unsigned int *)(v11 + 72);
  v90 = v11;
  v13 = v105;
  v103 = v105;
  if ( v12 )
  {
    v14 = v105;
    if ( v12 > 8 )
    {
      sub_2FB8E60((__int64)&v103, v12, v7, (__int64)v105, v9, v10);
      v14 = v103;
      v13 = &v103[96 * (unsigned int)v104];
    }
    for ( i = &v14[96 * v12]; i != v13; v13 += 96 )
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = 0;
        *((_QWORD *)v13 + 1) = v13 + 32;
        *((_DWORD *)v13 + 4) = 8;
        *((_DWORD *)v13 + 5) = 0;
        *((_DWORD *)v13 + 6) = 0;
        v13[28] = 1;
      }
    }
    LODWORD(v104) = v12;
  }
  v16 = *(__int64 **)(v8 + 64);
  v98 = 0;
  s = &v102;
  v17 = *(unsigned int *)(v8 + 72);
  *(_QWORD *)v100 = 8;
  v18 = &v16[v17];
  v101 = 1;
  *(_DWORD *)&v100[8] = 0;
  if ( v16 != v18 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v19 = *v16;
        v20 = *(_QWORD *)(*v16 + 8);
        if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          break;
LABEL_18:
        if ( v18 == ++v16 )
          goto LABEL_19;
      }
      v96 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
      v21 = sub_2E09D00((__int64 *)v96, v20);
      v24 = v21;
      if ( v21 == *(_QWORD *)v96 + 24LL * *(unsigned int *)(v96 + 8)
        || (v25 = *(_DWORD *)((*(_QWORD *)v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v21 >> 1) & 3,
            (unsigned int)v25 > (*(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((v20 >> 1) & 3))) )
      {
        BUG();
      }
      v26 = (__int64)&v103[96 * **(unsigned int **)(v21 + 16)];
      if ( !*(_BYTE *)(v26 + 28) )
        goto LABEL_149;
      v27 = *(_QWORD **)(v26 + 8);
      v24 = *(unsigned int *)(v26 + 20);
      v25 = (__int64)&v27[v24];
      if ( v27 != (_QWORD *)v25 )
      {
        while ( v19 != *v27 )
        {
          if ( (_QWORD *)v25 == ++v27 )
            goto LABEL_151;
        }
        goto LABEL_18;
      }
LABEL_151:
      if ( (unsigned int)v24 >= *(_DWORD *)(v26 + 16) )
      {
LABEL_149:
        ++v16;
        sub_C8CC70(v26, v19, v25, v24, v22, v23);
        if ( v18 == v16 )
          goto LABEL_19;
      }
      else
      {
        ++v16;
        *(_DWORD *)(v26 + 20) = v24 + 1;
        *(_QWORD *)v25 = v19;
        ++*(_QWORD *)v26;
        if ( v18 == v16 )
        {
LABEL_19:
          v91 = *(_DWORD *)(v90 + 72);
          if ( !v91 )
            goto LABEL_72;
          goto LABEL_20;
        }
      }
    }
  }
  v91 = *(_DWORD *)(v90 + 72);
  if ( !v91 )
    goto LABEL_74;
LABEL_20:
  v95 = a1;
  for ( j = 0; j != v91; ++j )
  {
    v28 = *(_DWORD *)(a2 + 24);
    v29 = *(_QWORD *)(a2 + 8);
    v97 = *(int **)(*(_QWORD *)(v90 + 64) + 8LL * j);
    if ( !v28 )
      continue;
    v30 = **(unsigned int **)(*(_QWORD *)(v90 + 64) + 8LL * j);
    v31 = v28 - 1;
    v32 = v31 & (37 * v30);
    v33 = *(_DWORD *)(v29 + 4LL * v32);
    if ( (_DWORD)v30 != v33 )
    {
      for ( k = 1; ; ++k )
      {
        if ( v33 == -1 )
          goto LABEL_71;
        v32 = v31 & (k + v32);
        v33 = *(_DWORD *)(v29 + 4LL * v32);
        if ( (_DWORD)v30 == v33 )
          break;
      }
    }
    v34 = v103;
    v35 = &v103[96 * v30];
    v36 = v35[28];
    v37 = (__int64 *)*((_QWORD *)v35 + 1);
    if ( v36 )
      v38 = *((unsigned int *)v35 + 5);
    else
      v38 = *((unsigned int *)v35 + 4);
    v39 = &v37[v38];
    for ( m = v37; m != v39; ++m )
    {
      if ( (unsigned __int64)*m < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
    v41 = &v103[96 * v30];
    if ( v36 )
      goto LABEL_29;
    while ( 2 )
    {
      if ( m == &v37[*((unsigned int *)v41 + 4)] )
        goto LABEL_51;
LABEL_30:
      v42 = m + 1;
      if ( m + 1 == v39 )
      {
        v43 = v39;
      }
      else
      {
        v43 = m + 1;
        do
        {
          if ( *v43 != -2 && *v43 != -1 )
            break;
          ++v43;
        }
        while ( v39 != v43 );
      }
      while ( 1 )
      {
        v44 = &v34[96 * v30];
        if ( !v36 )
          break;
LABEL_35:
        if ( v43 == &v37[*((unsigned int *)v44 + 5)] )
          goto LABEL_46;
LABEL_36:
        v45 = *m;
        if ( !v101 )
        {
          if ( !sub_C8CA60((__int64)&v98, v45) )
          {
            v63 = *v43;
            if ( v101 )
            {
              v46 = (char *)s;
              v47 = (char *)s + 8 * *(unsigned int *)&v100[4];
LABEL_84:
              if ( v47 != v46 )
              {
                while ( *(_QWORD *)v46 != v63 )
                {
                  v46 += 8;
                  if ( v46 == v47 )
                    goto LABEL_93;
                }
                goto LABEL_88;
              }
LABEL_93:
              v65 = *m;
              v66 = *(_QWORD *)(*(_QWORD *)(v95 + 8) + 32LL);
              v67 = *(_QWORD *)(*m + 8) & 0xFFFFFFFFFFFFFFF8LL;
              v68 = *(_QWORD *)(v67 + 16);
              if ( v68 )
              {
                v69 = *(_QWORD *)(v68 + 24);
              }
              else
              {
                v81 = *(unsigned int *)(v66 + 304);
                v82 = *(_QWORD **)(v66 + 296);
                if ( *(_DWORD *)(v66 + 304) )
                {
                  v83 = *(_DWORD *)(v67 + 24) | (*(__int64 *)(*m + 8) >> 1) & 3;
                  do
                  {
                    while ( 1 )
                    {
                      v84 = v81 >> 1;
                      v85 = &v82[2 * (v81 >> 1)];
                      if ( v83 < (*(_DWORD *)((*v85 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v85 >> 1) & 3) )
                        break;
                      v82 = v85 + 2;
                      v81 = v81 - v84 - 1;
                      if ( v81 <= 0 )
                        goto LABEL_134;
                    }
                    v81 >>= 1;
                  }
                  while ( v84 > 0 );
                }
LABEL_134:
                v69 = *(v82 - 1);
              }
              v70 = *(_QWORD *)(v63 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              v71 = *(_QWORD *)(v70 + 16);
              if ( v71 )
              {
                v72 = *(_QWORD *)(v71 + 24);
              }
              else
              {
                v78 = *(unsigned int *)(v66 + 304);
                v79 = *(_QWORD **)(v66 + 296);
                if ( *(_DWORD *)(v66 + 304) )
                {
                  do
                  {
                    while ( 1 )
                    {
                      v66 = v78 >> 1;
                      v80 = &v79[2 * (v78 >> 1)];
                      if ( (*(_DWORD *)(v70 + 24) | (unsigned int)((*(__int64 *)(v63 + 8) >> 1) & 3)) < (*(_DWORD *)((*v80 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v80 >> 1) & 3) )
                        break;
                      v79 = v80 + 2;
                      v78 = v78 - v66 - 1;
                      if ( v78 <= 0 )
                        goto LABEL_128;
                    }
                    v78 >>= 1;
                  }
                  while ( v66 > 0 );
                }
LABEL_128:
                v72 = *(v79 - 1);
              }
              if ( v72 == v69 )
              {
                v74 = *(_DWORD *)((*(_QWORD *)(v65 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                    | (unsigned int)(*(__int64 *)(v65 + 8) >> 1) & 3;
                if ( (unsigned int)v74 >= (*(_DWORD *)((*(_QWORD *)(v63 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                         | (unsigned int)(*(__int64 *)(v63 + 8) >> 1) & 3) )
                  v63 = *m;
                if ( v101 )
                {
                  v76 = s;
                  v65 = *(unsigned int *)&v100[4];
                  v74 = (__int64)s + 8 * *(unsigned int *)&v100[4];
                  if ( s != (void *)v74 )
                  {
                    while ( v63 != *v76 )
                    {
                      if ( (_QWORD *)v74 == ++v76 )
                        goto LABEL_114;
                    }
                    goto LABEL_88;
                  }
                  goto LABEL_114;
                }
              }
              else
              {
                v94 = v72;
                v73 = sub_2E6D360(*(_QWORD *)(v95 + 32), v69, v72);
                v72 = v94;
                if ( v73 )
                {
                  v63 = *v43;
                  if ( v101 )
                  {
                    v75 = s;
                    v65 = *(unsigned int *)&v100[4];
                    v74 = (__int64)s + 8 * *(unsigned int *)&v100[4];
                    if ( s != (void *)v74 )
                    {
                      while ( v63 != *v75 )
                      {
                        if ( (_QWORD *)v74 == ++v75 )
                          goto LABEL_114;
                      }
                      goto LABEL_88;
                    }
LABEL_114:
                    if ( (unsigned int)v65 < *(_DWORD *)v100 )
                    {
                      *(_DWORD *)&v100[4] = v65 + 1;
                      *(_QWORD *)v74 = v63;
                      ++v98;
                      goto LABEL_88;
                    }
                  }
                }
                else
                {
                  if ( !(unsigned __int8)sub_2E6D360(*(_QWORD *)(v95 + 32), v94, v69) )
                    goto LABEL_88;
                  v63 = *m;
                  if ( v101 )
                  {
                    v77 = s;
                    v65 = *(unsigned int *)&v100[4];
                    v74 = (__int64)s + 8 * *(unsigned int *)&v100[4];
                    if ( s != (void *)v74 )
                    {
                      while ( v63 != *v77 )
                      {
                        if ( (_QWORD *)v74 == ++v77 )
                          goto LABEL_114;
                      }
                      goto LABEL_88;
                    }
                    goto LABEL_114;
                  }
                }
              }
              sub_C8CC70((__int64)&v98, v63, v74, v65, v72, v66);
              goto LABEL_88;
            }
            if ( !sub_C8CA60((__int64)&v98, v63) )
            {
              v63 = *v43;
              goto LABEL_93;
            }
          }
LABEL_88:
          v34 = v103;
          v30 = (unsigned int)*v97;
          v64 = &v103[96 * v30];
          v36 = v64[28];
          v37 = (__int64 *)*((_QWORD *)v64 + 1);
          goto LABEL_41;
        }
        v46 = (char *)s;
        v47 = (char *)s + 8 * *(unsigned int *)&v100[4];
        if ( s == v47 )
        {
LABEL_83:
          v63 = *v43;
          goto LABEL_84;
        }
        v48 = s;
        while ( v45 != *v48 )
        {
          if ( v47 == (char *)++v48 )
            goto LABEL_83;
        }
LABEL_41:
        if ( ++v43 != v39 )
        {
          while ( (unsigned __int64)*v43 >= 0xFFFFFFFFFFFFFFFELL )
          {
            if ( v39 == ++v43 )
            {
              v44 = &v34[96 * v30];
              if ( v36 )
                goto LABEL_35;
              goto LABEL_45;
            }
          }
        }
      }
LABEL_45:
      if ( v43 != &v37[*((unsigned int *)v44 + 4)] )
        goto LABEL_36;
LABEL_46:
      while ( v39 != v42 )
      {
        if ( (unsigned __int64)*v42 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v42;
      }
      m = v42;
      v41 = &v34[96 * v30];
      if ( !v36 )
        continue;
      break;
    }
LABEL_29:
    if ( m != &v37[*((unsigned int *)v41 + 5)] )
      goto LABEL_30;
LABEL_51:
    if ( *(_DWORD *)&v100[4] == *(_DWORD *)&v100[8] )
      continue;
    v49 = 0;
    sub_2FB7E60(v95, 0, v97);
    v52 = (char *)s;
    if ( v101 )
      v53 = (char *)s + 8 * *(unsigned int *)&v100[4];
    else
      v53 = (char *)s + 8 * *(unsigned int *)v100;
    if ( v53 == s )
    {
LABEL_57:
      v54 = *(unsigned int *)(a3 + 8);
      LODWORD(v55) = *(_DWORD *)(a3 + 8);
      goto LABEL_58;
    }
    while ( *(_QWORD *)v52 >= 0xFFFFFFFFFFFFFFFELL )
    {
      v52 += 8;
      if ( v52 == v53 )
        goto LABEL_57;
    }
    v55 = *(unsigned int *)(a3 + 8);
    if ( v53 == v52 )
    {
      v54 = *(unsigned int *)(a3 + 8);
LABEL_58:
      LODWORD(v56) = 0;
      if ( *(unsigned int *)(a3 + 12) < v54 )
        goto LABEL_59;
      goto LABEL_65;
    }
    v86 = v52;
    v56 = 0;
    while ( 1 )
    {
      v87 = v86 + 8;
      if ( v86 + 8 == v53 )
        break;
      while ( 1 )
      {
        v86 = v87;
        if ( *(_QWORD *)v87 < 0xFFFFFFFFFFFFFFFELL )
          break;
        v87 += 8;
        if ( v53 == v87 )
          goto LABEL_142;
      }
      ++v56;
      if ( v53 == v87 )
        goto LABEL_143;
    }
LABEL_142:
    ++v56;
LABEL_143:
    v54 = v55 + v56;
    if ( v55 + v56 <= (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v49 = (_QWORD *)(*(_QWORD *)a3 + 8 * v55);
      goto LABEL_60;
    }
LABEL_59:
    sub_C8D5F0(a3, (const void *)(a3 + 16), v54, 8u, v50, v51);
    v55 = *(unsigned int *)(a3 + 8);
    v49 = (_QWORD *)(*(_QWORD *)a3 + 8 * v55);
    if ( v52 != v53 )
    {
LABEL_60:
      v57 = *(_QWORD *)v52;
      do
      {
        v58 = v52 + 8;
        *v49++ = v57;
        if ( v52 + 8 == v53 )
          break;
        while ( 1 )
        {
          v57 = *(_QWORD *)v58;
          v52 = v58;
          if ( *(_QWORD *)v58 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v58 += 8;
          if ( v53 == v58 )
            goto LABEL_64;
        }
      }
      while ( v53 != v58 );
LABEL_64:
      LODWORD(v55) = *(_DWORD *)(a3 + 8);
    }
LABEL_65:
    ++v98;
    v59 = v101 == 0;
    *(_DWORD *)(a3 + 8) = v56 + v55;
    if ( v59 )
    {
      v60 = 4 * (*(_DWORD *)&v100[4] - *(_DWORD *)&v100[8]);
      if ( v60 < 0x20 )
        v60 = 32;
      if ( *(_DWORD *)v100 <= v60 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v100);
        goto LABEL_70;
      }
      sub_C8C990((__int64)&v98, (__int64)v49);
    }
    else
    {
LABEL_70:
      *(_QWORD *)&v100[4] = 0;
    }
LABEL_71:
    ;
  }
LABEL_72:
  if ( !v101 )
    _libc_free((unsigned __int64)s);
LABEL_74:
  v61 = v103;
  v62 = (unsigned __int64)&v103[96 * (unsigned int)v104];
  if ( v103 != (_BYTE *)v62 )
  {
    do
    {
      while ( 1 )
      {
        v62 -= 96LL;
        if ( !*(_BYTE *)(v62 + 28) )
          break;
        if ( v61 == (_BYTE *)v62 )
          goto LABEL_79;
      }
      _libc_free(*(_QWORD *)(v62 + 8));
    }
    while ( v61 != (_BYTE *)v62 );
LABEL_79:
    v62 = (unsigned __int64)v103;
  }
  if ( (_BYTE *)v62 != v105 )
    _libc_free(v62);
}
