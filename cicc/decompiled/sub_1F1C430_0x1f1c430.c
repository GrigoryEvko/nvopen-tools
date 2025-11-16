// Function: sub_1F1C430
// Address: 0x1f1c430
//
void __fastcall sub_1F1C430(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // r13
  int v8; // r15d
  unsigned __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 v11; // rbx
  __int64 v12; // rax
  _QWORD *v13; // r13
  unsigned __int64 v14; // r14
  _QWORD *i; // r14
  __int64 *v16; // r13
  __int64 *v17; // rbx
  __int64 v18; // r15
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 *v22; // rax
  __int64 *v23; // rsi
  unsigned int v24; // r8d
  __int64 *v25; // rcx
  int v26; // edx
  int v27; // esi
  __int64 v28; // r8
  int v29; // edi
  __int64 v30; // rax
  unsigned int v31; // edx
  int v32; // ecx
  _BYTE *v33; // rdi
  _BYTE *v34; // rdx
  __int64 *v35; // rsi
  __int64 *v36; // r8
  __int64 *v37; // r12
  __int64 *v38; // rdx
  _BYTE *v39; // rdx
  __int64 *v40; // rcx
  __int64 *v41; // r15
  __int64 *v42; // rbx
  _BYTE *v43; // rdx
  __int64 *v44; // rdx
  char *v45; // rcx
  __int64 v46; // r14
  char *v47; // rax
  char *v48; // r13
  char *v49; // rsi
  char *v50; // r14
  _BYTE *v51; // rdx
  __int64 v52; // r8
  __int64 *v53; // r13
  __int64 v54; // r14
  __int64 v55; // rax
  __int64 v56; // r13
  __int64 v57; // rsi
  __int64 *v58; // rax
  __int64 v59; // rsi
  char *v60; // rsi
  __int64 v61; // r9
  __int64 *v62; // rdx
  __int64 v63; // r13
  __int64 *v64; // rax
  __int64 *v65; // rdi
  unsigned int v66; // r8d
  __int64 *v67; // rcx
  __int64 *v68; // rsi
  __int64 *v69; // rcx
  _BYTE *v70; // rbx
  unsigned __int64 v71; // r12
  unsigned __int64 v72; // rdi
  int v73; // r8d
  int v74; // r9d
  _BYTE *v75; // rdi
  _BYTE *v76; // rdx
  char *v77; // r13
  char *v78; // rax
  __int64 v79; // r12
  char *v80; // rbx
  unsigned int v81; // eax
  __int64 v82; // rdx
  char *v83; // rax
  unsigned int v84; // ebx
  __int64 v85; // rax
  __int64 v86; // r8
  __int64 v87; // rsi
  _QWORD *v88; // rcx
  _QWORD *v89; // rdx
  __int64 v92; // [rsp+28h] [rbp-458h]
  int v93; // [rsp+30h] [rbp-450h]
  unsigned int v94; // [rsp+34h] [rbp-44Ch]
  __int64 *v95; // [rsp+48h] [rbp-438h]
  __int64 v96; // [rsp+50h] [rbp-430h]
  __int64 v97; // [rsp+58h] [rbp-428h]
  __int64 v98; // [rsp+60h] [rbp-420h]
  __int64 v99; // [rsp+68h] [rbp-418h]
  __int64 v101; // [rsp+78h] [rbp-408h]
  __int64 v102; // [rsp+78h] [rbp-408h]
  __int64 v103; // [rsp+78h] [rbp-408h]
  int *v104; // [rsp+80h] [rbp-400h]
  __int64 v105; // [rsp+88h] [rbp-3F8h]
  __int64 *v106; // [rsp+88h] [rbp-3F8h]
  __int64 v107; // [rsp+90h] [rbp-3F0h] BYREF
  _BYTE *v108; // [rsp+98h] [rbp-3E8h]
  void *s; // [rsp+A0h] [rbp-3E0h]
  _BYTE v110[12]; // [rsp+A8h] [rbp-3D8h]
  _BYTE v111[72]; // [rsp+B8h] [rbp-3C8h] BYREF
  _BYTE *v112; // [rsp+100h] [rbp-380h] BYREF
  __int64 v113; // [rsp+108h] [rbp-378h]
  _BYTE v114[880]; // [rsp+110h] [rbp-370h] BYREF

  v6 = a1[9];
  v7 = a1[2];
  v8 = *(_DWORD *)(**(_QWORD **)(v6 + 16) + 4LL * *(unsigned int *)(v6 + 64));
  v9 = *(unsigned int *)(v7 + 408);
  v10 = v8 & 0x7FFFFFFF;
  if ( (v8 & 0x7FFFFFFFu) >= (unsigned int)v9 || (v11 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * v10)) == 0 )
  {
    v84 = v10 + 1;
    if ( (unsigned int)v9 < v10 + 1 )
    {
      v86 = v84;
      if ( v84 < v9 )
      {
        *(_DWORD *)(v7 + 408) = v84;
      }
      else if ( v84 > v9 )
      {
        if ( v84 > (unsigned __int64)*(unsigned int *)(v7 + 412) )
        {
          sub_16CD150(v7 + 400, (const void *)(v7 + 416), v84, 8, v84, a6);
          v9 = *(unsigned int *)(v7 + 408);
          v86 = v84;
        }
        v85 = *(_QWORD *)(v7 + 400);
        v87 = *(_QWORD *)(v7 + 416);
        v88 = (_QWORD *)(v85 + 8 * v86);
        v89 = (_QWORD *)(v85 + 8 * v9);
        if ( v88 != v89 )
        {
          do
            *v89++ = v87;
          while ( v88 != v89 );
          v85 = *(_QWORD *)(v7 + 400);
        }
        *(_DWORD *)(v7 + 408) = v84;
        goto LABEL_171;
      }
    }
    v85 = *(_QWORD *)(v7 + 400);
LABEL_171:
    *(_QWORD *)(v85 + 8LL * (v8 & 0x7FFFFFFF)) = sub_1DBA290(v8);
    v11 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * (v8 & 0x7FFFFFFF));
    sub_1DBB110((_QWORD *)v7, v11);
    v6 = a1[9];
  }
  v12 = *(_QWORD *)(v6 + 8);
  v107 = 0;
  v13 = v114;
  v108 = v111;
  s = v111;
  *(_QWORD *)v110 = 8;
  *(_DWORD *)&v110[8] = 0;
  v14 = *(unsigned int *)(v12 + 72);
  v92 = v12;
  v112 = v114;
  v113 = 0x800000000LL;
  if ( (unsigned int)v14 > 8 )
  {
    sub_1F18F50((__int64)&v112, v14);
    v13 = v112;
  }
  LODWORD(v113) = v14;
  for ( i = &v13[13 * v14]; i != v13; v13 += 13 )
  {
    if ( v13 )
      sub_16CCCB0(v13, (__int64)(v13 + 5), (__int64)&v107);
  }
  if ( v108 != s )
    _libc_free((unsigned __int64)s);
  v107 = 0;
  v108 = v111;
  s = v111;
  *(_QWORD *)v110 = 8;
  *(_DWORD *)&v110[8] = 0;
  v16 = *(__int64 **)(v11 + 64);
  v17 = &v16[*(unsigned int *)(v11 + 72)];
  if ( v17 == v16 )
  {
    v93 = *(_DWORD *)(v92 + 72);
    if ( !v93 )
      goto LABEL_121;
LABEL_28:
    v94 = 0;
    while ( 1 )
    {
      v26 = *(_DWORD *)(a2 + 24);
      if ( !v26 )
        goto LABEL_118;
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a2 + 8);
      v29 = 1;
      v104 = *(int **)(*(_QWORD *)(v92 + 64) + 8LL * v94);
      v30 = (unsigned int)*v104;
      v31 = (v26 - 1) & (37 * v30);
      v32 = *(_DWORD *)(v28 + 4LL * v31);
      if ( (_DWORD)v30 != v32 )
      {
        while ( v32 != -1 )
        {
          v31 = v27 & (v29 + v31);
          v32 = *(_DWORD *)(v28 + 4LL * v31);
          if ( (_DWORD)v30 == v32 )
            goto LABEL_31;
          ++v29;
        }
        goto LABEL_118;
      }
LABEL_31:
      v33 = v112;
      v34 = &v112[104 * v30];
      v35 = (__int64 *)*((_QWORD *)v34 + 2);
      v36 = (__int64 *)*((_QWORD *)v34 + 1);
      if ( v35 == v36 )
      {
        v37 = &v35[*((unsigned int *)v34 + 7)];
        if ( v35 == v37 )
        {
LABEL_155:
          v106 = (__int64 *)*((_QWORD *)v34 + 2);
          goto LABEL_37;
        }
      }
      else
      {
        v37 = &v35[*((unsigned int *)v34 + 6)];
        if ( v35 == v37 )
          goto LABEL_155;
      }
      v38 = (__int64 *)*((_QWORD *)v34 + 2);
      do
      {
        if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v38;
      }
      while ( v37 != v38 );
      v106 = v38;
LABEL_37:
      v39 = &v33[104 * v30];
      if ( v36 != v35 )
      {
        v40 = v106;
        if ( v106 != &v35[*((unsigned int *)v39 + 6)] )
          break;
        goto LABEL_117;
      }
      v40 = v106;
      if ( v106 != &v36[*((unsigned int *)v39 + 7)] )
        break;
LABEL_117:
      if ( *(_DWORD *)&v110[4] == *(_DWORD *)&v110[8] )
        goto LABEL_118;
      sub_1F1B3E0((__int64)a1, 0, v104);
      v75 = s;
      v76 = v108;
      if ( s == v108 )
        v77 = (char *)s + 8 * *(unsigned int *)&v110[4];
      else
        v77 = (char *)s + 8 * *(unsigned int *)v110;
      if ( s != v77 )
      {
        v78 = (char *)s;
        while ( 1 )
        {
          v79 = *(_QWORD *)v78;
          v80 = v78;
          if ( *(_QWORD *)v78 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v78 += 8;
          if ( v77 == v78 )
            goto LABEL_145;
        }
        if ( v77 != v78 )
        {
          v82 = *(unsigned int *)(a3 + 8);
          if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v82 )
            goto LABEL_164;
          while ( 1 )
          {
            *(_QWORD *)(*(_QWORD *)a3 + 8 * v82) = v79;
            v82 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
            v83 = v80 + 8;
            *(_DWORD *)(a3 + 8) = v82;
            if ( v80 + 8 == v77 )
              break;
            while ( 1 )
            {
              v79 = *(_QWORD *)v83;
              v80 = v83;
              if ( *(_QWORD *)v83 < 0xFFFFFFFFFFFFFFFELL )
                break;
              v83 += 8;
              if ( v77 == v83 )
                goto LABEL_161;
            }
            if ( v77 == v83 )
              break;
            if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v82 )
            {
LABEL_164:
              sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v73, v74);
              v82 = *(unsigned int *)(a3 + 8);
            }
          }
LABEL_161:
          v75 = s;
          v76 = v108;
        }
      }
LABEL_145:
      ++v107;
      if ( v75 != v76 )
      {
        v81 = 4 * (*(_DWORD *)&v110[4] - *(_DWORD *)&v110[8]);
        if ( v81 < 0x20 )
          v81 = 32;
        if ( *(_DWORD *)v110 > v81 )
        {
          sub_16CC920((__int64)&v107);
          goto LABEL_118;
        }
        memset(v75, -1, 8LL * *(unsigned int *)v110);
      }
      *(_QWORD *)&v110[4] = 0;
LABEL_118:
      if ( ++v94 == v93 )
        goto LABEL_119;
    }
    if ( v40 + 1 == v37 )
    {
      v42 = v37;
    }
    else
    {
      v42 = v40 + 1;
      do
      {
        if ( (unsigned __int64)(*v42 + 2) > 1 )
          break;
        ++v42;
      }
      while ( v37 != v42 );
    }
    v41 = v40 + 1;
    while ( 1 )
    {
      v43 = &v33[104 * v30];
      if ( v35 == v36 )
      {
        if ( v42 == &v35[*((unsigned int *)v43 + 7)] )
        {
LABEL_75:
          while ( v37 != v41 )
          {
            if ( (unsigned __int64)*v41 < 0xFFFFFFFFFFFFFFFELL )
              break;
            ++v41;
          }
          v106 = v41;
          goto LABEL_37;
        }
      }
      else if ( v42 == &v35[*((unsigned int *)v43 + 6)] )
      {
        goto LABEL_75;
      }
      v44 = (__int64 *)v108;
      v45 = (char *)s;
      v46 = *v106;
      v47 = v108;
      if ( s == v108 )
      {
        v48 = (char *)s + 8 * *(unsigned int *)&v110[4];
        if ( s == v48 )
        {
          v50 = (char *)s;
          v49 = (char *)s;
        }
        else
        {
          v49 = (char *)s;
          do
          {
            if ( v46 == *(_QWORD *)v49 )
              break;
            v49 += 8;
          }
          while ( v48 != v49 );
          v50 = (char *)s + 8 * *(unsigned int *)&v110[4];
        }
      }
      else
      {
        v48 = (char *)s + 8 * *(unsigned int *)v110;
        v49 = (char *)sub_16CC9F0((__int64)&v107, *v106);
        if ( v46 == *(_QWORD *)v49 )
        {
          v44 = (__int64 *)v108;
          v45 = (char *)s;
          v47 = v108;
          if ( s == v108 )
            v50 = (char *)s + 8 * *(unsigned int *)&v110[4];
          else
            v50 = (char *)s + 8 * *(unsigned int *)v110;
        }
        else
        {
          v44 = (__int64 *)v108;
          v45 = (char *)s;
          v47 = v108;
          if ( s != v108 )
          {
            v50 = (char *)s + 8 * *(unsigned int *)v110;
            if ( v48 != v50 )
              goto LABEL_50;
            v52 = *v42;
            goto LABEL_64;
          }
          v49 = (char *)s + 8 * *(unsigned int *)&v110[4];
          v50 = v49;
        }
      }
      for ( ; v50 != v49; v49 += 8 )
      {
        if ( *(_QWORD *)v49 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      if ( v49 != v48 )
        goto LABEL_50;
      v52 = *v42;
      if ( v44 == (__int64 *)v45 )
      {
        v59 = 8LL * *(unsigned int *)&v110[4];
        if ( &v44[(unsigned __int64)v59 / 8] == v44 )
        {
LABEL_91:
          v47 = &v45[v59];
          v53 = (__int64 *)v45;
          v60 = &v45[v59];
        }
        else
        {
          while ( v52 != *(_QWORD *)v47 )
          {
            v47 += 8;
            if ( &v44[(unsigned __int64)v59 / 8] == (__int64 *)v47 )
              goto LABEL_91;
          }
          v53 = (__int64 *)v45;
          v60 = &v45[v59];
        }
        goto LABEL_86;
      }
LABEL_64:
      v101 = v52;
      v47 = (char *)sub_16CC9F0((__int64)&v107, v52);
      if ( *(_QWORD *)v47 == v101 )
      {
        v53 = (__int64 *)s;
        v44 = (__int64 *)v108;
        if ( s == v108 )
          v60 = (char *)s + 8 * *(unsigned int *)&v110[4];
        else
          v60 = (char *)s + 8 * *(unsigned int *)v110;
      }
      else
      {
        v53 = (__int64 *)s;
        v44 = (__int64 *)v108;
        if ( s != v108 )
        {
          v47 = (char *)s + 8 * *(unsigned int *)v110;
          goto LABEL_67;
        }
        v47 = (char *)s + 8 * *(unsigned int *)&v110[4];
        v60 = v47;
      }
LABEL_86:
      while ( v60 != v47 && *(_QWORD *)v47 >= 0xFFFFFFFFFFFFFFFELL )
        v47 += 8;
LABEL_67:
      if ( v47 != v50 )
        goto LABEL_50;
      v95 = v44;
      v96 = *v106;
      v102 = *(_QWORD *)(a1[2] + 272LL);
      v97 = *(_QWORD *)(*v106 + 8);
      v54 = sub_1DA9310(v102, v97);
      v98 = *v42;
      v99 = *(_QWORD *)(*v42 + 8);
      v55 = sub_1DA9310(v102, v99);
      if ( v55 == v54 )
      {
        v61 = v98;
        v62 = v95;
        if ( (*(_DWORD *)((v97 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v97 >> 1) & 3) >= (*(_DWORD *)((v99 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v99 >> 1)
                                                                                               & 3) )
          v61 = v96;
        if ( v95 != v53 )
        {
LABEL_95:
          sub_16CCBA0((__int64)&v107, v61);
          goto LABEL_50;
        }
        v68 = &v95[*(unsigned int *)&v110[4]];
        if ( v68 == v95 )
        {
LABEL_151:
          if ( *(_DWORD *)&v110[4] >= *(_DWORD *)v110 )
            goto LABEL_95;
          ++*(_DWORD *)&v110[4];
          *v68 = v61;
          ++v107;
        }
        else
        {
          v69 = 0;
          while ( v61 != *v62 )
          {
            if ( *v62 == -2 )
              v69 = v62;
            if ( v68 == ++v62 )
            {
              if ( !v69 )
                goto LABEL_151;
              *v69 = v61;
              --*(_DWORD *)&v110[8];
              ++v107;
              break;
            }
          }
        }
      }
      else
      {
        v103 = v55;
        v56 = a1[5];
        sub_1E06620(v56);
        if ( sub_1E05550(*(_QWORD *)(v56 + 1312), v54, v103) )
        {
          v57 = *v42;
          v58 = (__int64 *)v108;
          if ( s != v108 )
            goto LABEL_71;
          v65 = (__int64 *)&v108[8 * *(unsigned int *)&v110[4]];
          v66 = *(_DWORD *)&v110[4];
          if ( v108 != (_BYTE *)v65 )
          {
            v67 = 0;
            while ( v57 != *v58 )
            {
              if ( *v58 == -2 )
                v67 = v58;
              if ( v65 == ++v58 )
              {
                if ( v67 )
                  goto LABEL_106;
                goto LABEL_136;
              }
            }
            goto LABEL_50;
          }
        }
        else
        {
          v63 = a1[5];
          sub_1E06620(v63);
          if ( !sub_1E05550(*(_QWORD *)(v63 + 1312), v103, v54) )
            goto LABEL_50;
          v57 = *v106;
          v64 = (__int64 *)v108;
          if ( s != v108 )
            goto LABEL_71;
          v65 = (__int64 *)&v108[8 * *(unsigned int *)&v110[4]];
          v66 = *(_DWORD *)&v110[4];
          if ( v108 != (_BYTE *)v65 )
          {
            v67 = 0;
            while ( v57 != *v64 )
            {
              if ( *v64 == -2 )
                v67 = v64;
              if ( v65 == ++v64 )
              {
                if ( !v67 )
                  goto LABEL_136;
LABEL_106:
                *v67 = v57;
                --*(_DWORD *)&v110[8];
                ++v107;
                goto LABEL_50;
              }
            }
            goto LABEL_50;
          }
        }
LABEL_136:
        if ( v66 >= *(_DWORD *)v110 )
        {
LABEL_71:
          sub_16CCBA0((__int64)&v107, v57);
          goto LABEL_50;
        }
        *(_DWORD *)&v110[4] = v66 + 1;
        *v65 = v57;
        ++v107;
      }
LABEL_50:
      for ( ++v42; v37 != v42; ++v42 )
      {
        if ( (unsigned __int64)*v42 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      v33 = v112;
      v30 = (unsigned int)*v104;
      v51 = &v112[104 * v30];
      v35 = (__int64 *)*((_QWORD *)v51 + 2);
      v36 = (__int64 *)*((_QWORD *)v51 + 1);
    }
  }
  do
  {
LABEL_15:
    v18 = *v16;
    v19 = *(_QWORD *)(*v16 + 8);
    if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v105 = *(_QWORD *)(a1[9] + 8LL);
      v20 = sub_1DB3C70((__int64 *)v105, *(_QWORD *)(*v16 + 8));
      if ( v20 == *(_QWORD *)v105 + 24LL * *(unsigned int *)(v105 + 8)
        || (*(_DWORD *)((*(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*(__int64 *)v20 >> 1) & 3)) > (*(_DWORD *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v19 >> 1) & 3) )
      {
        BUG();
      }
      v21 = (__int64)&v112[104 * **(unsigned int **)(v20 + 16)];
      v22 = *(__int64 **)(v21 + 8);
      if ( *(__int64 **)(v21 + 16) != v22 )
      {
LABEL_13:
        sub_16CCBA0(v21, v18);
        goto LABEL_14;
      }
      v23 = &v22[*(unsigned int *)(v21 + 28)];
      v24 = *(_DWORD *)(v21 + 28);
      if ( v22 == v23 )
      {
LABEL_181:
        if ( v24 >= *(_DWORD *)(v21 + 24) )
          goto LABEL_13;
        *(_DWORD *)(v21 + 28) = v24 + 1;
        *v23 = v18;
        ++*(_QWORD *)v21;
      }
      else
      {
        v25 = 0;
        while ( v18 != *v22 )
        {
          if ( *v22 == -2 )
            v25 = v22;
          if ( v23 == ++v22 )
          {
            if ( !v25 )
              goto LABEL_181;
            ++v16;
            *v25 = v18;
            --*(_DWORD *)(v21 + 32);
            ++*(_QWORD *)v21;
            if ( v17 != v16 )
              goto LABEL_15;
            goto LABEL_27;
          }
        }
      }
    }
LABEL_14:
    ++v16;
  }
  while ( v17 != v16 );
LABEL_27:
  v93 = *(_DWORD *)(v92 + 72);
  if ( v93 )
    goto LABEL_28;
LABEL_119:
  if ( s != v108 )
    _libc_free((unsigned __int64)s);
LABEL_121:
  v70 = v112;
  v71 = (unsigned __int64)&v112[104 * (unsigned int)v113];
  if ( v112 != (_BYTE *)v71 )
  {
    do
    {
      v71 -= 104LL;
      v72 = *(_QWORD *)(v71 + 16);
      if ( v72 != *(_QWORD *)(v71 + 8) )
        _libc_free(v72);
    }
    while ( v70 != (_BYTE *)v71 );
    v71 = (unsigned __int64)v112;
  }
  if ( (_BYTE *)v71 != v114 )
    _libc_free(v71);
}
