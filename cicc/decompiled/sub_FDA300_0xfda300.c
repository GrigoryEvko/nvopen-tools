// Function: sub_FDA300
// Address: 0xfda300
//
__int64 __fastcall sub_FDA300(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // r14
  int v10; // eax
  int v11; // r12d
  __int64 *v12; // r11
  int v13; // edx
  __int64 *v14; // rdi
  __int64 v15; // r8
  unsigned int v16; // edx
  __int64 v17; // rdi
  int v18; // eax
  __int64 v19; // r12
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned int v22; // ecx
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  void *v26; // rax
  _BYTE *v27; // rdi
  unsigned int v28; // ebx
  __int64 v30; // r9
  __int64 v31; // rdi
  __int64 v32; // rdx
  char *v33; // rdx
  __int64 v34; // r13
  char *v35; // rax
  __int64 v36; // rcx
  size_t v37; // rdx
  int v38; // eax
  __int64 *v39; // r10
  int v40; // ecx
  int v41; // edx
  __int64 v42; // rax
  __int64 *v43; // r12
  __int64 *v44; // r13
  int v45; // eax
  __int64 *v46; // rdi
  __int64 v47; // rcx
  int v48; // edx
  __int64 *v49; // r10
  __int64 v50; // rdi
  int v51; // eax
  int v52; // r11d
  __int64 *v53; // rcx
  unsigned int v54; // edx
  __int64 v55; // rdi
  _QWORD *v56; // rdi
  __int64 v57; // r8
  unsigned int v58; // eax
  int v59; // eax
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rax
  int v62; // ebx
  __int64 v63; // r12
  _QWORD *v64; // rax
  _QWORD *i; // rdx
  _QWORD *v66; // r8
  int v67; // r11d
  __int64 *v68; // r8
  __int64 *v69; // rcx
  unsigned int v70; // r12d
  unsigned int v71; // [rsp+0h] [rbp-C0h]
  __int64 v72; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v73; // [rsp+28h] [rbp-98h]
  __int64 v74; // [rsp+30h] [rbp-90h]
  __int64 v75; // [rsp+38h] [rbp-88h]
  void *src; // [rsp+40h] [rbp-80h] BYREF
  __int64 v77; // [rsp+48h] [rbp-78h]
  _BYTE v78[112]; // [rsp+50h] [rbp-70h] BYREF

  src = v78;
  v7 = *(__int64 **)(a2 + 24);
  v77 = 0x800000000LL;
  v8 = *(unsigned int *)(a2 + 32);
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v9 = &v7[6 * v8];
  v10 = 0;
  v75 = 0;
  if ( v9 == v7 )
  {
    v72 = 1;
    goto LABEL_23;
  }
  while ( v10 )
  {
    a2 = (unsigned int)v75;
    if ( !(_DWORD)v75 )
    {
      ++v72;
      goto LABEL_8;
    }
    v11 = 1;
    v12 = 0;
    a6 = (__int64)v73;
    v13 = (v75 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
    v14 = &v73[v13];
    v15 = *v14;
    if ( *v14 == *v7 )
    {
LABEL_4:
      v7 += 6;
      if ( v9 == v7 )
        goto LABEL_16;
    }
    else
    {
      while ( v15 != -4096 )
      {
        if ( v15 != -8192 || v12 )
          v14 = v12;
        v13 = (v75 - 1) & (v11 + v13);
        v15 = v73[v13];
        if ( *v7 == v15 )
          goto LABEL_4;
        ++v11;
        v12 = v14;
        v14 = &v73[v13];
      }
      if ( !v12 )
        v12 = v14;
      v18 = v10 + 1;
      ++v72;
      if ( 4 * v18 < (unsigned int)(3 * v75) )
      {
        if ( (int)v75 - HIDWORD(v74) - v18 <= (unsigned int)v75 >> 3 )
        {
          sub_BD14B0((__int64)&v72, v75);
          if ( !(_DWORD)v75 )
          {
LABEL_153:
            LODWORD(v74) = v74 + 1;
            BUG();
          }
          v15 = *v7;
          a6 = (__int64)v73;
          v39 = 0;
          v40 = 1;
          v41 = (v75 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
          v12 = &v73[v41];
          a2 = *v12;
          v18 = v74 + 1;
          if ( *v12 != *v7 )
          {
            while ( a2 != -4096 )
            {
              if ( a2 == -8192 && !v39 )
                v39 = v12;
              v41 = (v75 - 1) & (v40 + v41);
              v12 = &v73[v41];
              a2 = *v12;
              if ( v15 == *v12 )
                goto LABEL_10;
              ++v40;
            }
            if ( v39 )
              v12 = v39;
          }
        }
        goto LABEL_10;
      }
LABEL_8:
      a2 = (unsigned int)(2 * v75);
      sub_BD14B0((__int64)&v72, a2);
      if ( !(_DWORD)v75 )
        goto LABEL_153;
      a6 = *v7;
      v15 = (unsigned int)(v75 - 1);
      v16 = v15 & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
      v12 = &v73[v16];
      v17 = *v12;
      v18 = v74 + 1;
      if ( *v7 != *v12 )
      {
        a2 = 1;
        v69 = 0;
        while ( v17 != -4096 )
        {
          if ( !v69 && v17 == -8192 )
            v69 = v12;
          v70 = a2 + 1;
          v16 = v15 & (a2 + v16);
          a2 = v16;
          v12 = &v73[v16];
          v17 = *v12;
          if ( a6 == *v12 )
            goto LABEL_10;
          a2 = v70;
        }
        if ( v69 )
          v12 = v69;
      }
LABEL_10:
      LODWORD(v74) = v18;
      if ( *v12 != -4096 )
        --HIDWORD(v74);
      v19 = *v7;
      *v12 = *v7;
      v20 = (unsigned int)v77;
      v21 = (unsigned int)v77 + 1LL;
      if ( v21 > HIDWORD(v77) )
      {
        a2 = (__int64)v78;
        sub_C8D5F0((__int64)&src, v78, v21, 8u, v15, a6);
        v20 = (unsigned int)v77;
      }
      *((_QWORD *)src + v20) = v19;
      LODWORD(v77) = v77 + 1;
LABEL_15:
      v7 += 6;
      v10 = v74;
      if ( v9 == v7 )
      {
LABEL_16:
        ++v72;
        if ( !v10 )
          goto LABEL_38;
        v22 = 4 * v10;
        a2 = 64;
        v23 = (unsigned int)v75;
        if ( (unsigned int)(4 * v10) < 0x40 )
          v22 = 64;
        if ( (unsigned int)v75 <= v22 )
          goto LABEL_20;
        v56 = v73;
        v57 = (unsigned int)v75;
        v58 = v10 - 1;
        if ( v58 )
        {
          _BitScanReverse(&v58, v58);
          v59 = 1 << (33 - (v58 ^ 0x1F));
          if ( v59 < 64 )
            v59 = 64;
          if ( (_DWORD)v75 == v59 )
          {
            v74 = 0;
            v66 = &v73[v57];
            do
            {
              if ( v56 )
                *v56 = -4096;
              ++v56;
            }
            while ( v66 != v56 );
            goto LABEL_23;
          }
          v60 = (4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1);
          v61 = ((v60 | (v60 >> 2)) >> 4) | v60 | (v60 >> 2) | ((((v60 | (v60 >> 2)) >> 4) | v60 | (v60 >> 2)) >> 8);
          v62 = (v61 | (v61 >> 16)) + 1;
          v63 = 8 * ((v61 | (v61 >> 16)) + 1);
        }
        else
        {
          v63 = 1024;
          v62 = 128;
        }
        sub_C7D6A0((__int64)v73, v57 * 8, 8);
        a2 = 8;
        LODWORD(v75) = v62;
        v64 = (_QWORD *)sub_C7D670(v63, 8);
        v74 = 0;
        v73 = v64;
        for ( i = &v64[(unsigned int)v75]; i != v64; ++v64 )
        {
          if ( v64 )
            *v64 = -4096;
        }
        goto LABEL_23;
      }
    }
  }
  v30 = 8LL * (unsigned int)v77;
  a2 = (__int64)src + v30;
  v31 = v30 >> 5;
  v32 = v30 >> 5;
  a6 = v30 >> 3;
  v33 = (char *)src + 32 * v32;
  do
  {
    v34 = *v7;
    v35 = (char *)src;
    v36 = a6;
    if ( v31 )
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v35 == v34 )
          goto LABEL_35;
        if ( *((_QWORD *)v35 + 1) == v34 )
        {
          v35 += 8;
          goto LABEL_35;
        }
        if ( *((_QWORD *)v35 + 2) == v34 )
        {
          v35 += 16;
          goto LABEL_35;
        }
        if ( *((_QWORD *)v35 + 3) == v34 )
          break;
        v35 += 32;
        if ( v33 == v35 )
        {
          v36 = (a2 - (__int64)v33) >> 3;
          goto LABEL_62;
        }
      }
      v35 += 24;
      goto LABEL_35;
    }
LABEL_62:
    if ( v36 == 2 )
      goto LABEL_84;
    if ( v36 == 3 )
    {
      if ( *(_QWORD *)v35 == v34 )
        goto LABEL_35;
      v35 += 8;
LABEL_84:
      if ( *(_QWORD *)v35 == v34 )
        goto LABEL_35;
      v35 += 8;
      goto LABEL_65;
    }
    if ( v36 != 1 )
      goto LABEL_66;
LABEL_65:
    if ( *(_QWORD *)v35 != v34 )
    {
LABEL_66:
      if ( (unsigned __int64)(unsigned int)v77 + 1 > HIDWORD(v77) )
      {
        sub_C8D5F0((__int64)&src, v78, (unsigned int)v77 + 1LL, 8u, (unsigned int)v77, a6);
        a2 = (__int64)src + 8 * (unsigned int)v77;
      }
      *(_QWORD *)a2 = v34;
      v42 = (unsigned int)(v77 + 1);
      LODWORD(v77) = v42;
      if ( (unsigned int)v42 <= 8 )
        goto LABEL_15;
      v43 = (__int64 *)src;
      v44 = (__int64 *)((char *)src + 8 * v42);
      while ( 1 )
      {
        a2 = (unsigned int)v75;
        if ( !(_DWORD)v75 )
          break;
        a6 = (unsigned int)(v75 - 1);
        v45 = a6 & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
        v46 = &v73[v45];
        v47 = *v46;
        if ( *v43 != *v46 )
        {
          v52 = 1;
          v49 = 0;
          while ( v47 != -4096 )
          {
            if ( v49 || v47 != -8192 )
              v46 = v49;
            v45 = a6 & (v52 + v45);
            v47 = v73[v45];
            if ( *v43 == v47 )
              goto LABEL_71;
            ++v52;
            v49 = v46;
            v46 = &v73[v45];
          }
          if ( !v49 )
            v49 = v46;
          ++v72;
          v51 = v74 + 1;
          if ( 4 * ((int)v74 + 1) < (unsigned int)(3 * v75) )
          {
            if ( (int)v75 - HIDWORD(v74) - v51 <= (unsigned int)v75 >> 3 )
            {
              sub_BD14B0((__int64)&v72, v75);
              if ( !(_DWORD)v75 )
              {
LABEL_152:
                LODWORD(v74) = v74 + 1;
                BUG();
              }
              a2 = 1;
              v53 = 0;
              a6 = (__int64)v73;
              v54 = (v75 - 1) & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
              v49 = &v73[v54];
              v55 = *v49;
              v51 = v74 + 1;
              if ( *v43 != *v49 )
              {
                while ( v55 != -4096 )
                {
                  if ( !v53 && v55 == -8192 )
                    v53 = v49;
                  v54 = (v75 - 1) & (a2 + v54);
                  v71 = a2 + 1;
                  a2 = v54;
                  v49 = &v73[v54];
                  v55 = *v49;
                  if ( *v43 == *v49 )
                    goto LABEL_76;
                  a2 = v71;
                }
                if ( v53 )
                  v49 = v53;
              }
            }
            goto LABEL_76;
          }
LABEL_74:
          sub_BD14B0((__int64)&v72, 2 * v75);
          if ( !(_DWORD)v75 )
            goto LABEL_152;
          a2 = *v43;
          a6 = (__int64)v73;
          v48 = (v75 - 1) & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
          v49 = &v73[v48];
          v50 = *v49;
          v51 = v74 + 1;
          if ( *v43 != *v49 )
          {
            v67 = 1;
            v68 = 0;
            while ( v50 != -4096 )
            {
              if ( !v68 && v50 == -8192 )
                v68 = v49;
              v48 = (v75 - 1) & (v67 + v48);
              v49 = &v73[v48];
              v50 = *v49;
              if ( a2 == *v49 )
                goto LABEL_76;
              ++v67;
            }
            if ( v68 )
              v49 = v68;
          }
LABEL_76:
          LODWORD(v74) = v51;
          if ( *v49 != -4096 )
            --HIDWORD(v74);
          *v49 = *v43;
        }
LABEL_71:
        if ( v44 == ++v43 )
          goto LABEL_15;
      }
      ++v72;
      goto LABEL_74;
    }
LABEL_35:
    if ( (char *)a2 == v35 )
      goto LABEL_66;
    v7 += 6;
  }
  while ( v9 != v7 );
  ++v72;
LABEL_38:
  if ( HIDWORD(v74) )
  {
    v23 = (unsigned int)v75;
    if ( (unsigned int)v75 > 0x40 )
    {
      a2 = 8LL * (unsigned int)v75;
      sub_C7D6A0((__int64)v73, a2, 8);
      v73 = 0;
      v74 = 0;
      LODWORD(v75) = 0;
      goto LABEL_23;
    }
LABEL_20:
    v24 = v73;
    v25 = &v73[v23];
    if ( v73 != v25 )
    {
      do
        *v24++ = -4096;
      while ( v25 != v24 );
    }
    v74 = 0;
  }
LABEL_23:
  v26 = (void *)(a1 + 16);
  v27 = src;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v28 = v77;
  *(_QWORD *)a1 = a1 + 16;
  if ( v28 )
  {
    if ( v27 == v78 )
    {
      v37 = 8LL * v28;
      if ( v28 <= 8
        || (a2 = a1 + 16,
            sub_C8D5F0(a1, (const void *)(a1 + 16), v28, 8u, v28, a6),
            v26 = *(void **)a1,
            v27 = src,
            (v37 = 8LL * (unsigned int)v77) != 0) )
      {
        a2 = (__int64)v27;
        memcpy(v26, v27, v37);
        v27 = src;
      }
      *(_DWORD *)(a1 + 8) = v28;
      goto LABEL_24;
    }
    v38 = HIDWORD(v77);
    *(_QWORD *)a1 = v27;
    *(_DWORD *)(a1 + 8) = v28;
    *(_DWORD *)(a1 + 12) = v38;
  }
  else
  {
LABEL_24:
    if ( v27 != v78 )
      _libc_free(v27, a2);
  }
  sub_C7D6A0((__int64)v73, 8LL * (unsigned int)v75, 8);
  return a1;
}
