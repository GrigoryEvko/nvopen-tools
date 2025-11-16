// Function: sub_29385B0
// Address: 0x29385b0
//
__int16 __fastcall sub_29385B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned __int64 i; // r12
  __int64 v10; // rax
  void *v11; // rdx
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rsi
  char v17; // r14
  unsigned int j; // eax
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  void *v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  int v30; // r12d
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *k; // rdx
  __int64 v38; // r8
  int v39; // edx
  int v40; // ecx
  unsigned int v41; // edx
  __int64 *v42; // rdi
  __int64 v43; // r9
  __int16 v44; // ax
  char v45; // r14
  unsigned int v46; // eax
  __int64 v47; // rdx
  __int64 v48; // rax
  _QWORD *v49; // rdi
  __int64 v50; // rax
  __int64 v51; // r12
  char v52; // al
  _QWORD *v53; // rax
  __int64 v54; // rax
  __int64 *v55; // r13
  __int64 *v56; // r14
  _QWORD *v57; // rdi
  unsigned __int64 *v58; // rax
  unsigned __int64 v59; // r13
  __int64 v60; // rax
  __int64 *v61; // r13
  __int64 *v62; // r14
  _QWORD *v63; // rdi
  unsigned __int64 *v64; // rax
  unsigned __int64 v65; // r13
  __int64 v66; // r13
  __int64 v67; // rdx
  __int64 v68; // r15
  unsigned __int8 *v69; // r12
  __int64 v70; // rax
  __int64 v71; // r9
  __int64 v72; // rdx
  __int64 *v73; // r8
  unsigned __int64 v74; // rcx
  int v75; // eax
  unsigned __int64 *v76; // rdi
  __int64 v77; // rax
  unsigned int v78; // eax
  unsigned int v79; // ecx
  unsigned int v80; // eax
  _QWORD *v81; // rdi
  __int64 v82; // r12
  _QWORD *v83; // rax
  char *v84; // r12
  __int16 result; // ax
  unsigned __int64 v86; // rdx
  unsigned __int64 v87; // rax
  _QWORD *v88; // rax
  __int64 v89; // rdx
  _QWORD *m; // rdx
  int v91; // edi
  int v92; // r10d
  __int64 n; // rbx
  __int64 v94; // rdi
  char v96; // [rsp+14h] [rbp-ACh]
  char v97; // [rsp+17h] [rbp-A9h]
  __int64 v98; // [rsp+18h] [rbp-A8h]
  __int64 v99; // [rsp+20h] [rbp-A0h]
  __int64 v100; // [rsp+28h] [rbp-98h]
  __int16 v101; // [rsp+28h] [rbp-98h]
  __int64 v102; // [rsp+30h] [rbp-90h] BYREF
  __int64 v103; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int8 *v104; // [rsp+40h] [rbp-80h]
  __int64 v105; // [rsp+50h] [rbp-70h] BYREF
  void *s; // [rsp+58h] [rbp-68h]
  _BYTE v107[12]; // [rsp+60h] [rbp-60h]
  char v108; // [rsp+6Ch] [rbp-54h]
  char v109; // [rsp+70h] [rbp-50h] BYREF

  v6 = sub_B2BEC0(a2);
  v7 = *(_QWORD *)(a2 + 80);
  if ( !v7 )
    BUG();
  v8 = *(_QWORD *)(v7 + 32);
  for ( i = *(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF8LL; v8 != i; v8 = *(_QWORD *)(v8 + 8) )
  {
    if ( !v8 )
      BUG();
    if ( *(_BYTE *)(v8 - 24) == 60 )
    {
      v102 = v8 - 24;
      v100 = *(_QWORD *)(v8 + 48);
      sub_AE5020(v6, v100);
      v10 = sub_9208B0(v6, v100);
      s = v11;
      v105 = v10;
      if ( (_BYTE)v11 && (unsigned __int8)sub_2A4D8A0(v102, 0) )
        sub_2916C30(a1 + 600, &v102, v12, v13, v14, v15);
      else
        sub_2928360(a1 + 40, &v102);
    }
  }
  v16 = (__int64)&v102;
  v105 = 0;
  v17 = 0;
  *(_QWORD *)v107 = 4;
  *(_DWORD *)&v107[8] = 0;
  v108 = 1;
  v96 = 0;
  s = &v109;
  for ( j = *(_DWORD *)(a1 + 80); ; j = *(_DWORD *)(a1 + 80) )
  {
    if ( j )
      goto LABEL_38;
    v19 = *(unsigned int *)(a1 + 768);
    if ( (_DWORD)v19 )
    {
      if ( !(_BYTE)qword_5005428 )
      {
        sub_FFD350(*(_QWORD *)(a1 + 8), v16, v19, v3, v4, v5);
        v16 = *(unsigned int *)(a1 + 768);
        sub_2A57B70(*(void **)(a1 + 760));
      }
      ++*(_QWORD *)(a1 + 600);
      if ( !*(_BYTE *)(a1 + 628) )
      {
        v20 = 4 * (*(_DWORD *)(a1 + 620) - *(_DWORD *)(a1 + 624));
        v21 = *(unsigned int *)(a1 + 616);
        if ( v20 < 0x20 )
          v20 = 32;
        if ( v20 < (unsigned int)v21 )
        {
          sub_C8C990(a1 + 600, v16);
          goto LABEL_22;
        }
        memset(*(void **)(a1 + 608), -1, 8 * v21);
      }
      *(_QWORD *)(a1 + 620) = 0;
LABEL_22:
      *(_DWORD *)(a1 + 768) = 0;
      v17 = 1;
    }
    v22 = 8LL * *(unsigned int *)(a1 + 64);
    sub_C7D6A0(*(_QWORD *)(a1 + 48), v22, 8);
    v25 = *(unsigned int *)(a1 + 448);
    *(_DWORD *)(a1 + 64) = v25;
    if ( (_DWORD)v25 )
    {
      v26 = (void *)sub_C7D670(8 * v25, 8);
      v27 = *(unsigned int *)(a1 + 64);
      v22 = *(_QWORD *)(a1 + 432);
      *(_QWORD *)(a1 + 48) = v26;
      *(_QWORD *)(a1 + 56) = *(_QWORD *)(a1 + 440);
      memcpy(v26, (const void *)v22, 8 * v27);
    }
    else
    {
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
    }
    v28 = *(unsigned int *)(a1 + 464);
    v29 = *(unsigned int *)(a1 + 80);
    v30 = *(_DWORD *)(a1 + 464);
    if ( v28 <= v29 )
    {
      if ( *(_DWORD *)(a1 + 464) )
      {
        v22 = *(_QWORD *)(a1 + 456);
        memmove(*(void **)(a1 + 72), (const void *)v22, 8 * v28);
      }
    }
    else
    {
      if ( v28 > *(unsigned int *)(a1 + 84) )
      {
        v31 = 0;
        *(_DWORD *)(a1 + 80) = 0;
        sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v28, 8u, v23, v24);
        v28 = *(unsigned int *)(a1 + 464);
      }
      else
      {
        v31 = 8 * v29;
        if ( *(_DWORD *)(a1 + 80) )
        {
          memmove(*(void **)(a1 + 72), *(const void **)(a1 + 456), 8 * v29);
          v28 = *(unsigned int *)(a1 + 464);
        }
      }
      v32 = *(_QWORD *)(a1 + 456);
      v33 = 8 * v28;
      v22 = v32 + v31;
      if ( v32 + v31 != v33 + v32 )
        memcpy((void *)(v31 + *(_QWORD *)(a1 + 72)), (const void *)v22, v33 - v31);
    }
    v34 = *(_DWORD *)(a1 + 440);
    ++*(_QWORD *)(a1 + 424);
    *(_DWORD *)(a1 + 80) = v30;
    if ( !v34 )
    {
      if ( !*(_DWORD *)(a1 + 444) )
        goto LABEL_37;
      v35 = *(unsigned int *)(a1 + 448);
      if ( (unsigned int)v35 > 0x40 )
      {
        v22 = 8 * v35;
        sub_C7D6A0(*(_QWORD *)(a1 + 432), 8 * v35, 8);
        *(_QWORD *)(a1 + 432) = 0;
        *(_QWORD *)(a1 + 440) = 0;
        *(_DWORD *)(a1 + 448) = 0;
        goto LABEL_37;
      }
LABEL_34:
      v36 = *(_QWORD **)(a1 + 432);
      for ( k = &v36[v35]; k != v36; ++v36 )
        *v36 = -4096;
      *(_QWORD *)(a1 + 440) = 0;
      goto LABEL_37;
    }
    v79 = 4 * v34;
    v22 = 64;
    v35 = *(unsigned int *)(a1 + 448);
    if ( (unsigned int)(4 * v34) < 0x40 )
      v79 = 64;
    if ( v79 >= (unsigned int)v35 )
      goto LABEL_34;
    v80 = v34 - 1;
    if ( v80 )
    {
      _BitScanReverse(&v80, v80);
      v81 = *(_QWORD **)(a1 + 432);
      v82 = (unsigned int)(1 << (33 - (v80 ^ 0x1F)));
      if ( (int)v82 < 64 )
        v82 = 64;
      if ( (_DWORD)v82 == (_DWORD)v35 )
      {
        *(_QWORD *)(a1 + 440) = 0;
        v83 = &v81[v82];
        do
        {
          if ( v81 )
            *v81 = -4096;
          ++v81;
        }
        while ( v83 != v81 );
        goto LABEL_37;
      }
    }
    else
    {
      v81 = *(_QWORD **)(a1 + 432);
      LODWORD(v82) = 64;
    }
    sub_C7D6A0((__int64)v81, 8 * v35, 8);
    v22 = 8;
    v86 = ((((((((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v82 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v82 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v82 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v82 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 16;
    v87 = (v86
         | (((((((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v82 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v82 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v82 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v82 / 3u + 1) | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v82 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v82 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 448) = v87;
    v88 = (_QWORD *)sub_C7D670(8 * v87, 8);
    v89 = *(unsigned int *)(a1 + 448);
    *(_QWORD *)(a1 + 440) = 0;
    *(_QWORD *)(a1 + 432) = v88;
    for ( m = &v88[v89]; m != v88; ++v88 )
    {
      if ( v88 )
        *v88 = -4096;
    }
LABEL_37:
    *(_DWORD *)(a1 + 464) = 0;
    j = *(_DWORD *)(a1 + 80);
    if ( !j )
      break;
LABEL_38:
    v38 = *(_QWORD *)(a1 + 48);
    v16 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * j - 8);
    v39 = *(_DWORD *)(a1 + 64);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = (v39 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v42 = (__int64 *)(v38 + 8LL * v41);
      v43 = *v42;
      if ( v16 == *v42 )
      {
LABEL_40:
        *v42 = -8192;
        j = *(_DWORD *)(a1 + 80);
        --*(_DWORD *)(a1 + 56);
        ++*(_DWORD *)(a1 + 60);
      }
      else
      {
        v91 = 1;
        while ( v43 != -4096 )
        {
          v92 = v91 + 1;
          v41 = v40 & (v91 + v41);
          v42 = (__int64 *)(v38 + 8LL * v41);
          v43 = *v42;
          if ( v16 == *v42 )
            goto LABEL_40;
          v91 = v92;
        }
      }
    }
    *(_DWORD *)(a1 + 80) = j - 1;
    v44 = sub_2935C30(a1, (_QWORD *)v16);
    v96 |= HIBYTE(v44);
    v97 = v44 | v17;
    v45 = 0;
LABEL_42:
    v46 = *(_DWORD *)(a1 + 224);
    if ( v46 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)(a1 + 216);
        v102 = 4;
        v103 = 0;
        v16 = v3 + 24LL * v46 - 24;
        v104 = *(unsigned __int8 **)(v16 + 16);
        v47 = (__int64)v104;
        LOBYTE(v4) = v104 != 0;
        if ( v104 + 4096 != 0 && v104 != 0 && v104 != (unsigned __int8 *)-8192LL )
        {
          v16 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050((unsigned __int64 *)&v102, v16);
          v46 = *(_DWORD *)(a1 + 224);
          v3 = *(_QWORD *)(a1 + 216);
        }
        v48 = v46 - 1;
        *(_DWORD *)(a1 + 224) = v48;
        v49 = (_QWORD *)(v3 + 24 * v48);
        v50 = v49[2];
        LOBYTE(v3) = v50 != 0;
        if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
          sub_BD60C0(v49);
        v51 = (__int64)v104;
        if ( !v104 )
          goto LABEL_42;
        LOBYTE(v47) = v104 + 0x2000 != 0;
        v52 = v47 & (v104 + 4096 != 0);
        if ( *v104 <= 0x1Cu )
        {
          if ( v52 )
            sub_BD60C0(&v102);
          goto LABEL_42;
        }
        if ( v52 )
          sub_BD60C0(&v102);
        if ( *(_BYTE *)v51 == 60 )
          break;
LABEL_81:
        v66 = v51;
        sub_AE94E0(v51);
        v16 = sub_ACA8A0(*(__int64 ***)(v51 + 8));
        sub_BD84D0(v51, v16);
        if ( (*(_BYTE *)(v51 + 7) & 0x40) != 0 )
        {
          v67 = *(_QWORD *)(v51 - 8);
          v66 = v67 + 32LL * (*(_DWORD *)(v51 + 4) & 0x7FFFFFF);
        }
        else
        {
          v67 = v51 - 32LL * (*(_DWORD *)(v51 + 4) & 0x7FFFFFF);
        }
        v68 = v67;
        v98 = a1 + 216;
        if ( v66 != v67 )
        {
          v99 = v51;
          do
          {
            while ( 1 )
            {
              v69 = *(unsigned __int8 **)v68;
              if ( **(_BYTE **)v68 > 0x1Cu )
              {
                v70 = *(_QWORD *)(v68 + 8);
                **(_QWORD **)(v68 + 16) = v70;
                if ( v70 )
                  *(_QWORD *)(v70 + 16) = *(_QWORD *)(v68 + 16);
                *(_QWORD *)v68 = 0;
                v16 = 0;
                if ( sub_F50EE0(v69, 0) )
                {
                  v102 = 4;
                  v103 = 0;
                  v104 = v69;
                  if ( v69 != (unsigned __int8 *)-8192LL && v69 != (unsigned __int8 *)-4096LL )
                    sub_BD73F0((__int64)&v102);
                  v72 = *(unsigned int *)(a1 + 224);
                  v73 = &v102;
                  v74 = *(_QWORD *)(a1 + 216);
                  v16 = v72 + 1;
                  v75 = *(_DWORD *)(a1 + 224);
                  if ( v72 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 228) )
                  {
                    if ( v74 > (unsigned __int64)&v102 || (unsigned __int64)&v102 >= v74 + 24 * v72 )
                    {
                      sub_D6B130(v98, v16, v72, v74, (__int64)&v102, v71);
                      v72 = *(unsigned int *)(a1 + 224);
                      v74 = *(_QWORD *)(a1 + 216);
                      v73 = &v102;
                      v75 = *(_DWORD *)(a1 + 224);
                    }
                    else
                    {
                      v84 = (char *)&v102 - v74;
                      sub_D6B130(v98, v16, v72, v74, (__int64)&v102, v71);
                      v74 = *(_QWORD *)(a1 + 216);
                      v72 = *(unsigned int *)(a1 + 224);
                      v73 = (__int64 *)&v84[v74];
                      v75 = *(_DWORD *)(a1 + 224);
                    }
                  }
                  v76 = (unsigned __int64 *)(v74 + 24 * v72);
                  if ( v76 )
                  {
                    *v76 = 4;
                    v77 = v73[2];
                    v76[1] = 0;
                    v76[2] = v77;
                    if ( v77 != 0 && v77 != -4096 && v77 != -8192 )
                    {
                      v16 = *v73 & 0xFFFFFFFFFFFFFFF8LL;
                      sub_BD6050(v76, v16);
                    }
                    v75 = *(_DWORD *)(a1 + 224);
                  }
                  *(_DWORD *)(a1 + 224) = v75 + 1;
                  if ( v104 + 4096 != 0 && v104 != 0 && v104 != (unsigned __int8 *)-8192LL )
                    break;
                }
              }
              v68 += 32;
              if ( v66 == v68 )
                goto LABEL_102;
            }
            v68 += 32;
            sub_BD60C0(&v102);
          }
          while ( v66 != v68 );
LABEL_102:
          v51 = v99;
        }
        v45 = 1;
        sub_B43D60((_QWORD *)v51);
        v46 = *(_DWORD *)(a1 + 224);
        if ( !v46 )
          goto LABEL_104;
      }
      if ( !v108 )
        goto LABEL_117;
      v53 = s;
      v3 = *(unsigned int *)&v107[4];
      v47 = (__int64)s + 8 * *(unsigned int *)&v107[4];
      if ( s == (void *)v47 )
      {
LABEL_130:
        if ( *(_DWORD *)&v107[4] < *(_DWORD *)v107 )
        {
          ++*(_DWORD *)&v107[4];
          *(_QWORD *)v47 = v51;
          ++v105;
          goto LABEL_59;
        }
LABEL_117:
        sub_C8CC70((__int64)&v105, v51, v47, v3, v4, v5);
        goto LABEL_59;
      }
      while ( v51 != *v53 )
      {
        if ( (_QWORD *)v47 == ++v53 )
          goto LABEL_130;
      }
LABEL_59:
      sub_AE74C0((unsigned __int64 *)&v102, v51);
      v54 = v102;
      if ( (v102 & 4) != 0 )
      {
        v55 = *(__int64 **)(v102 & 0xFFFFFFFFFFFFFFF8LL);
        v56 = &v55[*(unsigned int *)((v102 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
        if ( v55 != v56 )
        {
          do
          {
LABEL_62:
            v57 = (_QWORD *)*v55++;
            sub_B43D60(v57);
          }
          while ( v56 != v55 );
          v54 = v102;
        }
      }
      else if ( (v102 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v55 = &v102;
        v56 = &v103;
        goto LABEL_62;
      }
      if ( v54 )
      {
        if ( (v54 & 4) != 0 )
        {
          v58 = (unsigned __int64 *)(v54 & 0xFFFFFFFFFFFFFFF8LL);
          v59 = (unsigned __int64)v58;
          if ( v58 )
          {
            if ( (unsigned __int64 *)*v58 != v58 + 2 )
              _libc_free(*v58);
            j_j___libc_free_0(v59);
          }
        }
      }
      sub_AE7690((unsigned __int64 *)&v102, v51);
      v60 = v102;
      if ( (v102 & 4) != 0 )
      {
        v61 = *(__int64 **)(v102 & 0xFFFFFFFFFFFFFFF8LL);
        v62 = &v61[*(unsigned int *)((v102 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
        if ( v61 != v62 )
        {
          do
          {
LABEL_73:
            v63 = (_QWORD *)*v61++;
            sub_B14290(v63);
          }
          while ( v62 != v61 );
          v60 = v102;
        }
      }
      else if ( (v102 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v61 = &v102;
        v62 = &v103;
        goto LABEL_73;
      }
      if ( v60 )
      {
        if ( (v60 & 4) != 0 )
        {
          v64 = (unsigned __int64 *)(v60 & 0xFFFFFFFFFFFFFFF8LL);
          v65 = (unsigned __int64)v64;
          if ( v64 )
          {
            if ( (unsigned __int64 *)*v64 != v64 + 2 )
              _libc_free(*v64);
            j_j___libc_free_0(v65);
          }
        }
      }
      goto LABEL_81;
    }
LABEL_104:
    v17 = v97 | v45;
    if ( *(_DWORD *)&v107[4] != *(_DWORD *)&v107[8] )
    {
      sub_29214F0(a1 + 40, (__int64)&v105);
      sub_29214F0(a1 + 424, (__int64)&v105);
      v16 = (__int64)&v105;
      sub_2921310(a1 + 600, (__int64)&v105);
      ++v105;
      if ( v108 )
        goto LABEL_110;
      v78 = 4 * (*(_DWORD *)&v107[4] - *(_DWORD *)&v107[8]);
      if ( v78 < 0x20 )
        v78 = 32;
      if ( v78 >= *(_DWORD *)v107 )
      {
        v16 = 0xFFFFFFFFLL;
        memset(s, -1, 8LL * *(unsigned int *)v107);
LABEL_110:
        *(_QWORD *)&v107[4] = 0;
        continue;
      }
      sub_C8C990((__int64)&v105, (__int64)&v105);
    }
  }
  if ( v17 )
  {
    if ( (unsigned __int8)sub_AEA460(*(_QWORD *)(a2 + 40)) )
    {
      for ( n = *(_QWORD *)(a2 + 80); a2 + 72 != n; n = *(_QWORD *)(n + 8) )
      {
        v94 = n - 24;
        if ( !n )
          v94 = 0;
        sub_F3F2F0(v94, v22);
      }
    }
  }
  LOBYTE(result) = v17;
  HIBYTE(result) = v96;
  if ( !v108 )
  {
    v101 = result;
    _libc_free((unsigned __int64)s);
    return v101;
  }
  return result;
}
