// Function: sub_34F80E0
// Address: 0x34f80e0
//
void __fastcall sub_34F80E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned int v11; // edi
  unsigned int v12; // esi
  _QWORD *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r14
  __int64 *v24; // r12
  __int64 v25; // rbx
  __int64 *v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // rdi
  unsigned int v31; // edx
  __int64 v32; // rbx
  __int64 v33; // rdi
  unsigned int v34; // edx
  __int64 v35; // rcx
  __int64 v36; // rsi
  int v37; // r11d
  unsigned int v38; // r10d
  __int64 *v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // rcx
  unsigned int v42; // edx
  __int64 v43; // rbx
  int v44; // edi
  _QWORD *v45; // r11
  unsigned int v46; // r14d
  unsigned int v47; // r10d
  _QWORD *v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rdx
  unsigned int v53; // eax
  _QWORD *v54; // rax
  _QWORD *v55; // rcx
  int v56; // eax
  int v57; // edx
  __int64 *v58; // rax
  __int64 *v59; // rdx
  __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rbx
  __int64 v63; // rdi
  int v64; // r11d
  __int64 v65; // r10
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 *v69; // rax
  __int64 *v70; // r14
  __int64 *v71; // r13
  __int64 v72; // rcx
  __int64 *v73; // rax
  __int64 *v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // rdi
  int v77; // eax
  __int64 v78; // rdi
  __int64 v79; // rax
  __int64 v80; // rsi
  int v81; // r10d
  int v82; // eax
  int v83; // esi
  __int64 v84; // rdi
  unsigned int v85; // r14d
  __int64 v86; // rax
  int v87; // ecx
  int v88; // edx
  int v89; // eax
  int v90; // eax
  int v91; // r11d
  int v92; // r11d
  __int64 v93; // r10
  unsigned int v94; // ecx
  int v95; // edi
  int v96; // r10d
  int v97; // r10d
  unsigned int v98; // r13d
  __int64 v99; // rcx
  __int64 v100; // rdi
  int v101; // r10d
  int v102; // r10d
  _QWORD *v103; // rcx
  __int64 v104; // r14
  int v105; // r8d
  __int64 v106; // rsi
  int v107; // r11d
  int v108; // r11d
  __int64 v109; // r10
  unsigned int v110; // ecx
  __int64 v111; // rdi
  _QWORD *v112; // rsi
  unsigned int v113; // r10d
  unsigned int v114; // r8d
  _QWORD *v119; // [rsp+38h] [rbp-108h]
  _QWORD *v120; // [rsp+40h] [rbp-100h]
  __int64 v121; // [rsp+48h] [rbp-F8h]
  unsigned int v122; // [rsp+48h] [rbp-F8h]
  __int64 v123; // [rsp+50h] [rbp-F0h] BYREF
  __int64 *v124; // [rsp+58h] [rbp-E8h]
  __int64 v125; // [rsp+60h] [rbp-E0h]
  int v126; // [rsp+68h] [rbp-D8h]
  unsigned __int8 v127; // [rsp+6Ch] [rbp-D4h]
  char v128; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v129; // [rsp+B0h] [rbp-90h] BYREF
  void *s; // [rsp+B8h] [rbp-88h]
  _BYTE v131[12]; // [rsp+C0h] [rbp-80h]
  unsigned __int8 v132; // [rsp+CCh] [rbp-74h]
  char v133; // [rsp+D0h] [rbp-70h] BYREF

  v124 = (__int64 *)&v128;
  s = &v133;
  v8 = *(_QWORD *)(a1 + 32);
  v9 = v8;
  v123 = 0;
  v125 = 8;
  v126 = 0;
  v127 = 1;
  v129 = 0;
  *(_QWORD *)v131 = 8;
  *(_DWORD *)&v131[8] = 0;
  v132 = 1;
  if ( a2 )
  {
    v10 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v11 = *(_DWORD *)(a2 + 24) + 1;
  }
  else
  {
    v10 = 0;
    v11 = 0;
  }
  v12 = *(_DWORD *)(v8 + 32);
  if ( v12 <= v11 )
    BUG();
  v121 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v10) + 8LL);
  v13 = *(_QWORD **)(a3 + 8);
  if ( *(_BYTE *)(a3 + 28) )
    v14 = *(unsigned int *)(a3 + 20);
  else
    v14 = *(unsigned int *)(a3 + 16);
  v120 = &v13[v14];
  v15 = (__int64)v120;
  if ( v13 == v120 )
    goto LABEL_9;
  while ( 1 )
  {
    v16 = *v13;
    if ( *v13 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v120 == ++v13 )
      goto LABEL_9;
  }
  v119 = v13;
  if ( v120 == v13 )
  {
LABEL_9:
    v17 = a2;
    if ( !a2 )
      goto LABEL_67;
    goto LABEL_10;
  }
  while ( 2 )
  {
    v29 = *(_QWORD *)(v16 + 24);
    if ( v29 )
    {
      v30 = (unsigned int)(*(_DWORD *)(v29 + 24) + 1);
      v31 = *(_DWORD *)(v29 + 24) + 1;
    }
    else
    {
      v30 = 0;
      v31 = 0;
    }
    v32 = 0;
    if ( v31 < v12 )
      v32 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v30);
    while ( 1 )
    {
      if ( v121 == v32 )
        goto LABEL_83;
      if ( v29 )
      {
        v33 = (unsigned int)(*(_DWORD *)(v29 + 24) + 1);
        v34 = *(_DWORD *)(v29 + 24) + 1;
      }
      else
      {
        v33 = 0;
        v34 = 0;
      }
      v35 = 0;
      if ( v34 < v12 )
        v35 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v33);
      if ( v35 != v32 )
        break;
LABEL_77:
      if ( v127 )
      {
        v58 = v124;
        v59 = &v124[HIDWORD(v125)];
        if ( v124 != v59 )
        {
          while ( v32 != *v58 )
          {
            if ( v59 == ++v58 )
              goto LABEL_106;
          }
LABEL_82:
          v9 = *(_QWORD *)(a1 + 32);
          v12 = *(_DWORD *)(v9 + 32);
          goto LABEL_83;
        }
      }
      else if ( sub_C8CA60((__int64)&v123, v32) )
      {
        goto LABEL_82;
      }
LABEL_106:
      if ( v132 )
      {
        v75 = s;
        v35 = *(unsigned int *)&v131[4];
        v59 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v131[4]);
        if ( s == v59 )
        {
LABEL_116:
          if ( *(_DWORD *)&v131[4] >= *(_DWORD *)v131 )
            goto LABEL_112;
          ++*(_DWORD *)&v131[4];
          *v59 = v32;
          v8 = *(_QWORD *)(a1 + 32);
          ++v129;
          v32 = *(_QWORD *)(v32 + 8);
          v12 = *(_DWORD *)(v8 + 32);
          v9 = v8;
        }
        else
        {
          while ( v32 != *v75 )
          {
            if ( v59 == ++v75 )
              goto LABEL_116;
          }
          v8 = *(_QWORD *)(a1 + 32);
          v32 = *(_QWORD *)(v32 + 8);
          v12 = *(_DWORD *)(v8 + 32);
          v9 = v8;
        }
      }
      else
      {
LABEL_112:
        sub_C8CC70((__int64)&v129, v32, (__int64)v59, v35, v10, v15);
        v8 = *(_QWORD *)(a1 + 32);
        v32 = *(_QWORD *)(v32 + 8);
        v12 = *(_DWORD *)(v8 + 32);
        v9 = v8;
      }
    }
    v36 = *(unsigned int *)(a7 + 24);
    if ( !(_DWORD)v36 )
    {
      ++*(_QWORD *)a7;
LABEL_124:
      sub_34F7160(a7, 2 * v36);
      v77 = *(_DWORD *)(a7 + 24);
      if ( !v77 )
        goto LABEL_233;
      v10 = (unsigned int)(v77 - 1);
      v78 = *(_QWORD *)(a7 + 8);
      LODWORD(v79) = v10 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v57 = *(_DWORD *)(a7 + 16) + 1;
      v35 = v78 + 16LL * (unsigned int)v79;
      v80 = *(_QWORD *)v35;
      if ( v32 != *(_QWORD *)v35 )
      {
        v81 = 1;
        v15 = 0;
        while ( v80 != -4096 )
        {
          if ( !v15 && v80 == -8192 )
            v15 = v35;
          v79 = (unsigned int)v10 & ((_DWORD)v79 + v81);
          v35 = v78 + 16 * v79;
          v80 = *(_QWORD *)v35;
          if ( v32 == *(_QWORD *)v35 )
            goto LABEL_74;
          ++v81;
        }
        if ( v15 )
          v35 = v15;
      }
LABEL_74:
      *(_DWORD *)(a7 + 16) = v57;
      if ( *(_QWORD *)v35 != -4096 )
        --*(_DWORD *)(a7 + 20);
      *(_QWORD *)v35 = v32;
      *(_QWORD *)(v35 + 8) = 0;
      goto LABEL_77;
    }
    v15 = (unsigned int)(v36 - 1);
    v10 = *(_QWORD *)(a7 + 8);
    v37 = 1;
    v35 = 0;
    v38 = v15 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v39 = (__int64 *)(v10 + 16LL * v38);
    v40 = *v39;
    if ( v32 == *v39 )
      goto LABEL_47;
    while ( 1 )
    {
      if ( v40 == -4096 )
      {
        v56 = *(_DWORD *)(a7 + 16);
        if ( !v35 )
          v35 = (__int64)v39;
        ++*(_QWORD *)a7;
        v57 = v56 + 1;
        if ( 4 * (v56 + 1) >= (unsigned int)(3 * v36) )
          goto LABEL_124;
        if ( (int)v36 - *(_DWORD *)(a7 + 20) - v57 <= (unsigned int)v36 >> 3 )
        {
          sub_34F7160(a7, v36);
          v82 = *(_DWORD *)(a7 + 24);
          if ( !v82 )
          {
LABEL_233:
            ++*(_DWORD *)(a7 + 16);
            BUG();
          }
          v83 = v82 - 1;
          v84 = *(_QWORD *)(a7 + 8);
          v10 = 0;
          v85 = (v82 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
          v15 = 1;
          v57 = *(_DWORD *)(a7 + 16) + 1;
          v35 = v84 + 16LL * v85;
          v86 = *(_QWORD *)v35;
          if ( v32 != *(_QWORD *)v35 )
          {
            while ( v86 != -4096 )
            {
              if ( !v10 && v86 == -8192 )
                v10 = v35;
              v113 = v15 + 1;
              v15 = v83 & (v85 + (unsigned int)v15);
              v85 = v15;
              v35 = v84 + 16LL * (unsigned int)v15;
              v86 = *(_QWORD *)v35;
              if ( v32 == *(_QWORD *)v35 )
                goto LABEL_74;
              v15 = v113;
            }
            if ( v10 )
              v35 = v10;
          }
        }
        goto LABEL_74;
      }
      if ( v40 != -8192 || v35 )
        v39 = (__int64 *)v35;
      v35 = (unsigned int)(v37 + 1);
      v38 = v15 & (v37 + v38);
      v40 = *(_QWORD *)(v10 + 16LL * v38);
      if ( v32 == v40 )
        break;
      ++v37;
      v35 = (__int64)v39;
      v39 = (__int64 *)(v10 + 16LL * v38);
    }
    v39 = (__int64 *)(v10 + 16LL * v38);
LABEL_47:
    if ( !v39[1] )
      goto LABEL_77;
    v41 = 0;
    v42 = 0;
    if ( v29 )
    {
      v41 = (unsigned int)(*(_DWORD *)(v29 + 24) + 1);
      v42 = *(_DWORD *)(v29 + 24) + 1;
    }
    v43 = 0;
    if ( v42 < *(_DWORD *)(v8 + 32) )
      v43 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v41);
    v44 = 1;
    v45 = 0;
    v46 = ((unsigned int)v43 >> 4) ^ ((unsigned int)v43 >> 9);
    v47 = v46 & v15;
    v48 = (_QWORD *)(v10 + 16LL * (v46 & (unsigned int)v15));
    v49 = *v48;
    if ( v43 != *v48 )
    {
      while ( v49 != -4096 )
      {
        if ( !v45 && v49 == -8192 )
          v45 = v48;
        v47 = v15 & (v44 + v47);
        v48 = (_QWORD *)(v10 + 16LL * v47);
        v49 = *v48;
        if ( v43 == *v48 )
          goto LABEL_53;
        ++v44;
      }
      v89 = *(_DWORD *)(a7 + 16);
      if ( v45 )
        v48 = v45;
      ++*(_QWORD *)a7;
      v90 = v89 + 1;
      if ( 4 * v90 >= (unsigned int)(3 * v36) )
      {
        sub_34F7160(a7, 2 * v36);
        v107 = *(_DWORD *)(a7 + 24);
        if ( v107 )
        {
          v108 = v107 - 1;
          v109 = *(_QWORD *)(a7 + 8);
          v110 = v108 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v90 = *(_DWORD *)(a7 + 16) + 1;
          v48 = (_QWORD *)(v109 + 16LL * v110);
          v111 = *v48;
          if ( v43 != *v48 )
          {
            v15 = 1;
            v112 = 0;
            while ( v111 != -4096 )
            {
              if ( v111 == -8192 && !v112 )
                v112 = v48;
              v114 = v15 + 1;
              v15 = v110 + (unsigned int)v15;
              v110 = v108 & v15;
              v48 = (_QWORD *)(v109 + 16LL * (v108 & (unsigned int)v15));
              v111 = *v48;
              if ( v43 == *v48 )
                goto LABEL_170;
              v15 = v114;
            }
            if ( v112 )
              v48 = v112;
          }
          goto LABEL_170;
        }
      }
      else
      {
        if ( (int)v36 - *(_DWORD *)(a7 + 20) - v90 > (unsigned int)v36 >> 3 )
        {
LABEL_170:
          *(_DWORD *)(a7 + 16) = v90;
          if ( *v48 != -4096 )
            --*(_DWORD *)(a7 + 20);
          *v48 = v43;
          v48[1] = 0;
          v8 = *(_QWORD *)(a1 + 32);
          goto LABEL_173;
        }
        sub_34F7160(a7, v36);
        v101 = *(_DWORD *)(a7 + 24);
        if ( v101 )
        {
          v102 = v101 - 1;
          v15 = *(_QWORD *)(a7 + 8);
          v103 = 0;
          LODWORD(v104) = v102 & v46;
          v105 = 1;
          v90 = *(_DWORD *)(a7 + 16) + 1;
          v48 = (_QWORD *)(v15 + 16LL * (unsigned int)v104);
          v106 = *v48;
          if ( v43 != *v48 )
          {
            while ( v106 != -4096 )
            {
              if ( !v103 && v106 == -8192 )
                v103 = v48;
              v104 = v102 & (unsigned int)(v104 + v105);
              v48 = (_QWORD *)(v15 + 16 * v104);
              v106 = *v48;
              if ( v43 == *v48 )
                goto LABEL_170;
              ++v105;
            }
            if ( v103 )
              v48 = v103;
          }
          goto LABEL_170;
        }
      }
      ++*(_DWORD *)(a7 + 16);
      BUG();
    }
LABEL_53:
    v50 = v48[1];
    if ( v50 )
    {
      v51 = *(unsigned int *)(a5 + 8);
      if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        v36 = a5 + 16;
        sub_C8D5F0(a5, (const void *)(a5 + 16), v51 + 1, 8u, v10, v15);
        v51 = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v51) = v50;
      LOBYTE(v52) = v132;
      ++*(_DWORD *)(a5 + 8);
      goto LABEL_57;
    }
LABEL_173:
    v12 = *(_DWORD *)(v8 + 32);
    v9 = v8;
LABEL_83:
    if ( v29 )
    {
      v60 = (unsigned int)(*(_DWORD *)(v29 + 24) + 1);
      v61 = *(_DWORD *)(v29 + 24) + 1;
    }
    else
    {
      v60 = 0;
      v61 = 0;
    }
    v62 = 0;
    if ( v61 < v12 )
      v62 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 8 * v60);
    v36 = *(unsigned int *)(a6 + 24);
    if ( !(_DWORD)v36 )
    {
      ++*(_QWORD *)a6;
      goto LABEL_175;
    }
    v63 = *(_QWORD *)(a6 + 8);
    v10 = (unsigned int)(v36 - 1);
    v64 = 1;
    v65 = 0;
    LODWORD(v66) = v10 & (((unsigned int)v62 >> 4) ^ ((unsigned int)v62 >> 9));
    v67 = v63 + 16LL * (unsigned int)v66;
    v68 = *(_QWORD *)v67;
    if ( v62 == *(_QWORD *)v67 )
      goto LABEL_89;
    while ( v68 != -4096 )
    {
      if ( !v65 && v68 == -8192 )
        v65 = v67;
      v15 = (unsigned int)(v64 + 1);
      v66 = (unsigned int)v10 & ((_DWORD)v66 + v64);
      v67 = v63 + 16 * v66;
      v68 = *(_QWORD *)v67;
      if ( v62 == *(_QWORD *)v67 )
        goto LABEL_89;
      ++v64;
    }
    if ( v65 )
      v67 = v65;
    v87 = *(_DWORD *)(a6 + 16);
    ++*(_QWORD *)a6;
    v88 = v87 + 1;
    if ( 4 * (v87 + 1) >= (unsigned int)(3 * v36) )
    {
LABEL_175:
      v36 = (unsigned int)(2 * v36);
      sub_34F7F00(a6, v36);
      v91 = *(_DWORD *)(a6 + 24);
      if ( v91 )
      {
        v92 = v91 - 1;
        v93 = *(_QWORD *)(a6 + 8);
        v94 = v92 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v88 = *(_DWORD *)(a6 + 16) + 1;
        v67 = v93 + 16LL * v94;
        v10 = *(_QWORD *)v67;
        if ( v62 != *(_QWORD *)v67 )
        {
          v95 = 1;
          v36 = 0;
          while ( v10 != -4096 )
          {
            if ( v10 == -8192 && !v36 )
              v36 = v67;
            v15 = (unsigned int)(v95 + 1);
            v94 = v92 & (v95 + v94);
            v67 = v93 + 16LL * v94;
            v10 = *(_QWORD *)v67;
            if ( v62 == *(_QWORD *)v67 )
              goto LABEL_157;
            ++v95;
          }
          if ( v36 )
            v67 = v36;
        }
        goto LABEL_157;
      }
      goto LABEL_235;
    }
    if ( (int)v36 - *(_DWORD *)(a6 + 20) - v88 <= (unsigned int)v36 >> 3 )
    {
      sub_34F7F00(a6, v36);
      v96 = *(_DWORD *)(a6 + 24);
      if ( v96 )
      {
        v97 = v96 - 1;
        v15 = *(_QWORD *)(a6 + 8);
        v36 = 1;
        v98 = v97 & (((unsigned int)v62 >> 4) ^ ((unsigned int)v62 >> 9));
        v88 = *(_DWORD *)(a6 + 16) + 1;
        v99 = 0;
        v67 = v15 + 16LL * v98;
        v100 = *(_QWORD *)v67;
        if ( v62 != *(_QWORD *)v67 )
        {
          while ( v100 != -4096 )
          {
            if ( v100 == -8192 && !v99 )
              v99 = v67;
            v10 = (unsigned int)(v36 + 1);
            v98 = v97 & (v36 + v98);
            v67 = v15 + 16LL * v98;
            v100 = *(_QWORD *)v67;
            if ( v62 == *(_QWORD *)v67 )
              goto LABEL_157;
            v36 = (unsigned int)v10;
          }
          if ( v99 )
            v67 = v99;
        }
        goto LABEL_157;
      }
LABEL_235:
      ++*(_DWORD *)(a6 + 16);
      BUG();
    }
LABEL_157:
    *(_DWORD *)(a6 + 16) = v88;
    if ( *(_QWORD *)v67 != -4096 )
      --*(_DWORD *)(a6 + 20);
    *(_QWORD *)v67 = v62;
    *(_DWORD *)(v67 + 8) = 0;
LABEL_89:
    *(_DWORD *)(v67 + 8) = 0;
    v52 = v132;
    v69 = (__int64 *)s;
    if ( v132 )
      v70 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v131[4]);
    else
      v70 = (__int64 *)((char *)s + 8 * *(unsigned int *)v131);
    if ( s != v70 )
    {
      while ( 1 )
      {
        v36 = *v69;
        v71 = v69;
        if ( (unsigned __int64)*v69 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v70 == ++v69 )
          goto LABEL_57;
      }
      if ( v69 != v70 )
      {
        v72 = v127;
        if ( v127 )
        {
LABEL_97:
          v73 = v124;
          v52 = (__int64)&v124[HIDWORD(v125)];
          if ( v124 != (__int64 *)v52 )
          {
            do
            {
              if ( v36 == *v73 )
                goto LABEL_101;
              ++v73;
            }
            while ( (__int64 *)v52 != v73 );
          }
          if ( HIDWORD(v125) < (unsigned int)v125 )
          {
            ++HIDWORD(v125);
            *(_QWORD *)v52 = v36;
            v72 = v127;
            ++v123;
            goto LABEL_101;
          }
        }
        while ( 1 )
        {
          sub_C8CC70((__int64)&v123, v36, v52, v72, v10, v15);
          v72 = v127;
LABEL_101:
          v74 = v71 + 1;
          if ( v71 + 1 == v70 )
            break;
          while ( 1 )
          {
            v36 = *v74;
            v71 = v74;
            if ( (unsigned __int64)*v74 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v70 == ++v74 )
              goto LABEL_104;
          }
          if ( v74 == v70 )
            break;
          if ( (_BYTE)v72 )
            goto LABEL_97;
        }
LABEL_104:
        LOBYTE(v52) = v132;
      }
    }
LABEL_57:
    ++v129;
    if ( (_BYTE)v52 )
    {
LABEL_62:
      *(_QWORD *)&v131[4] = 0;
    }
    else
    {
      v53 = 4 * (*(_DWORD *)&v131[4] - *(_DWORD *)&v131[8]);
      if ( v53 < 0x20 )
        v53 = 32;
      if ( v53 >= *(_DWORD *)v131 )
      {
        memset(s, -1, 8LL * *(unsigned int *)v131);
        goto LABEL_62;
      }
      sub_C8C990((__int64)&v129, v36);
    }
    v54 = v119 + 1;
    if ( v119 + 1 != v120 )
    {
      while ( 1 )
      {
        v16 = *v54;
        v55 = v54;
        if ( *v54 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v120 == ++v54 )
          goto LABEL_66;
      }
      v8 = *(_QWORD *)(a1 + 32);
      v119 = v55;
      v76 = v55;
      v12 = *(_DWORD *)(v8 + 32);
      v9 = v8;
      if ( v120 == v76 )
        goto LABEL_9;
      continue;
    }
    break;
  }
LABEL_66:
  v9 = *(_QWORD *)(a1 + 32);
  v17 = a2;
  v12 = *(_DWORD *)(v9 + 32);
  if ( a2 )
  {
LABEL_10:
    v18 = (unsigned int)(*(_DWORD *)(v17 + 24) + 1);
    v19 = *(_DWORD *)(v17 + 24) + 1;
    goto LABEL_11;
  }
LABEL_67:
  v18 = 0;
  v19 = 0;
LABEL_11:
  v20 = 0;
  if ( v19 < v12 )
    v20 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 8 * v18);
  v21 = *(unsigned int *)(a4 + 8);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v21 + 1, 8u, v10, v15);
    v21 = *(unsigned int *)(a4 + 8);
  }
  v122 = 0;
  *(_QWORD *)(*(_QWORD *)a4 + 8 * v21) = v20;
  ++*(_DWORD *)(a4 + 8);
  while ( 2 )
  {
    v22 = *(_QWORD *)(*(_QWORD *)a4 + 8LL * v122++);
    v23 = *(__int64 **)(v22 + 24);
    v24 = &v23[*(unsigned int *)(v22 + 32)];
    if ( v23 != v24 )
    {
      while ( 2 )
      {
        v25 = *v23;
        if ( v127 )
        {
          v26 = v124;
          v27 = &v124[HIDWORD(v125)];
          if ( v124 == v27 )
            goto LABEL_25;
          while ( v25 != *v26 )
          {
            if ( v27 == ++v26 )
              goto LABEL_25;
          }
        }
        else if ( !sub_C8CA60((__int64)&v123, v25) )
        {
          goto LABEL_25;
        }
        v28 = *(unsigned int *)(a4 + 8);
        if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          sub_C8D5F0(a4, (const void *)(a4 + 16), v28 + 1, 8u, v10, v15);
          v28 = *(unsigned int *)(a4 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * v28) = v25;
        ++*(_DWORD *)(a4 + 8);
LABEL_25:
        if ( v24 == ++v23 )
          break;
        continue;
      }
    }
    if ( v122 != *(_DWORD *)(a4 + 8) )
      continue;
    break;
  }
  if ( !v132 )
    _libc_free((unsigned __int64)s);
  if ( !v127 )
    _libc_free((unsigned __int64)v124);
}
