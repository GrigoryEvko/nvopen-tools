// Function: sub_21F15B0
// Address: 0x21f15b0
//
void __fastcall sub_21F15B0(__int64 a1)
{
  int v1; // r15d
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 v6; // r8
  unsigned int v7; // edi
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r9
  unsigned int v19; // esi
  int v20; // edx
  int v21; // edx
  __int64 v22; // r8
  __int64 v23; // rsi
  int v24; // ecx
  __int64 v25; // rdi
  int v26; // eax
  __int64 v27; // rdx
  _DWORD *v28; // rax
  _DWORD *i; // rdx
  __int64 v30; // rax
  int v31; // r14d
  _QWORD *v32; // rbx
  unsigned int v33; // eax
  __int64 v34; // rdx
  _QWORD *v35; // r13
  __int64 v36; // r14
  int v37; // edx
  int v38; // r11d
  __int64 *v39; // r10
  int v40; // ecx
  int v41; // esi
  int v42; // esi
  int v43; // r10d
  __int64 *v44; // r9
  __int64 v45; // r8
  unsigned int v46; // edx
  __int64 v47; // rdi
  unsigned int v48; // ecx
  _DWORD *v49; // rdi
  unsigned int v50; // eax
  int v51; // eax
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rax
  int v54; // ebx
  __int64 v55; // r13
  _DWORD *v56; // rax
  __int64 v57; // rdx
  _DWORD *j; // rdx
  __int64 v59; // r15
  int v60; // esi
  unsigned int v61; // edx
  int v62; // ebx
  unsigned int v63; // r14d
  unsigned int v64; // eax
  _QWORD *v65; // rdi
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rdi
  _QWORD *v68; // rax
  __int64 v69; // rdx
  _QWORD *k; // rdx
  int v71; // r10d
  _DWORD *v72; // rax
  _QWORD *v73; // rax
  unsigned int v74; // [rsp+Ch] [rbp-64h]
  __int64 v75; // [rsp+10h] [rbp-60h]
  __int64 v76; // [rsp+18h] [rbp-58h]
  __int64 v77; // [rsp+20h] [rbp-50h] BYREF
  __int64 v78; // [rsp+28h] [rbp-48h]
  __int64 v79; // [rsp+30h] [rbp-40h]

  v1 = 0;
  v3 = *(_QWORD *)a1;
  v77 = 0;
  v78 = 0;
  v4 = *(_QWORD *)(v3 + 328);
  v79 = 0;
  sub_1DFC3F0((__int64)&v77, v4);
  v5 = v78;
  v75 = a1 + 144;
  v76 = v77;
  if ( v77 != v78 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_QWORD *)(v5 - 8);
      sub_1E06620(v10);
      v12 = *(_QWORD *)(v10 + 1312);
      v13 = 0;
      v14 = *(unsigned int *)(v12 + 48);
      if ( !(_DWORD)v14 )
        goto LABEL_9;
      v15 = *(_QWORD *)(v12 + 32);
      v16 = (v14 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v11 == *v17 )
      {
LABEL_7:
        if ( v17 != (__int64 *)(v15 + 16 * v14) )
        {
          v13 = v17[1];
          goto LABEL_9;
        }
      }
      else
      {
        v37 = 1;
        while ( v18 != -8 )
        {
          v60 = v37 + 1;
          v16 = (v14 - 1) & (v37 + v16);
          v17 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( v11 == *v17 )
            goto LABEL_7;
          v37 = v60;
        }
      }
      v13 = 0;
LABEL_9:
      v19 = *(_DWORD *)(a1 + 168);
      ++v1;
      if ( !v19 )
      {
        ++*(_QWORD *)(a1 + 144);
        goto LABEL_11;
      }
      v6 = *(_QWORD *)(a1 + 152);
      v7 = (v19 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( v13 == *v8 )
      {
LABEL_4:
        *((_DWORD *)v8 + 2) = v1;
        v5 -= 8;
        if ( v76 == v5 )
          break;
      }
      else
      {
        v38 = 1;
        v39 = 0;
        while ( v9 != -8 )
        {
          if ( v9 == -16 && !v39 )
            v39 = v8;
          v7 = (v19 - 1) & (v38 + v7);
          v8 = (__int64 *)(v6 + 16LL * v7);
          v9 = *v8;
          if ( v13 == *v8 )
            goto LABEL_4;
          ++v38;
        }
        v40 = *(_DWORD *)(a1 + 160);
        if ( v39 )
          v8 = v39;
        ++*(_QWORD *)(a1 + 144);
        v24 = v40 + 1;
        if ( 4 * v24 < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 164) - v24 > v19 >> 3 )
            goto LABEL_13;
          v74 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
          sub_1E20790(v75, v19);
          v41 = *(_DWORD *)(a1 + 168);
          if ( !v41 )
          {
LABEL_118:
            ++*(_DWORD *)(a1 + 160);
            BUG();
          }
          v42 = v41 - 1;
          v43 = 1;
          v44 = 0;
          v45 = *(_QWORD *)(a1 + 152);
          v46 = v42 & v74;
          v24 = *(_DWORD *)(a1 + 160) + 1;
          v8 = (__int64 *)(v45 + 16LL * (v42 & v74));
          v47 = *v8;
          if ( v13 == *v8 )
            goto LABEL_13;
          while ( v47 != -8 )
          {
            if ( !v44 && v47 == -16 )
              v44 = v8;
            v46 = v42 & (v43 + v46);
            v8 = (__int64 *)(v45 + 16LL * v46);
            v47 = *v8;
            if ( v13 == *v8 )
              goto LABEL_13;
            ++v43;
          }
          goto LABEL_53;
        }
LABEL_11:
        sub_1E20790(v75, 2 * v19);
        v20 = *(_DWORD *)(a1 + 168);
        if ( !v20 )
          goto LABEL_118;
        v21 = v20 - 1;
        v22 = *(_QWORD *)(a1 + 152);
        LODWORD(v23) = v21 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v24 = *(_DWORD *)(a1 + 160) + 1;
        v8 = (__int64 *)(v22 + 16LL * (unsigned int)v23);
        v25 = *v8;
        if ( v13 == *v8 )
          goto LABEL_13;
        v71 = 1;
        v44 = 0;
        while ( v25 != -8 )
        {
          if ( v25 == -16 && !v44 )
            v44 = v8;
          v23 = v21 & (unsigned int)(v23 + v71);
          v8 = (__int64 *)(v22 + 16 * v23);
          v25 = *v8;
          if ( v13 == *v8 )
            goto LABEL_13;
          ++v71;
        }
LABEL_53:
        if ( v44 )
          v8 = v44;
LABEL_13:
        *(_DWORD *)(a1 + 160) = v24;
        if ( *v8 != -8 )
          --*(_DWORD *)(a1 + 164);
        *((_DWORD *)v8 + 2) = 0;
        v5 -= 8;
        *v8 = v13;
        *((_DWORD *)v8 + 2) = v1;
        if ( v76 == v5 )
          break;
      }
    }
  }
  v26 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  if ( !v26 )
  {
    if ( !*(_DWORD *)(a1 + 76) )
      goto LABEL_22;
    v27 = *(unsigned int *)(a1 + 80);
    if ( (unsigned int)v27 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 64));
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 80) = 0;
      goto LABEL_22;
    }
    goto LABEL_19;
  }
  v48 = 4 * v26;
  v27 = *(unsigned int *)(a1 + 80);
  if ( (unsigned int)(4 * v26) < 0x40 )
    v48 = 64;
  if ( v48 >= (unsigned int)v27 )
  {
LABEL_19:
    v28 = *(_DWORD **)(a1 + 64);
    for ( i = &v28[2 * v27]; i != v28; v28 += 2 )
      *v28 = -1;
    *(_QWORD *)(a1 + 72) = 0;
    goto LABEL_22;
  }
  v49 = *(_DWORD **)(a1 + 64);
  v50 = v26 - 1;
  if ( !v50 )
  {
    v55 = 1024;
    v54 = 128;
LABEL_64:
    j___libc_free_0(v49);
    *(_DWORD *)(a1 + 80) = v54;
    v56 = (_DWORD *)sub_22077B0(v55);
    v57 = *(unsigned int *)(a1 + 80);
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 64) = v56;
    for ( j = &v56[2 * v57]; j != v56; v56 += 2 )
    {
      if ( v56 )
        *v56 = -1;
    }
    goto LABEL_22;
  }
  _BitScanReverse(&v50, v50);
  v51 = 1 << (33 - (v50 ^ 0x1F));
  if ( v51 < 64 )
    v51 = 64;
  if ( (_DWORD)v27 != v51 )
  {
    v52 = (((4 * v51 / 3u + 1) | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)) >> 2)
        | (4 * v51 / 3u + 1)
        | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)
        | (((((4 * v51 / 3u + 1) | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)) >> 2)
          | (4 * v51 / 3u + 1)
          | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)) >> 4);
    v53 = (v52 >> 8) | v52;
    v54 = (v53 | (v53 >> 16)) + 1;
    v55 = 8 * ((v53 | (v53 >> 16)) + 1);
    goto LABEL_64;
  }
  *(_QWORD *)(a1 + 72) = 0;
  v72 = &v49[2 * v27];
  do
  {
    if ( v49 )
      *v49 = -1;
    v49 += 2;
  }
  while ( v72 != v49 );
LABEL_22:
  v30 = *(_QWORD *)(a1 + 88);
  if ( v30 != *(_QWORD *)(a1 + 96) )
    *(_QWORD *)(a1 + 96) = v30;
  v31 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v31 || *(_DWORD *)(a1 + 132) )
  {
    v32 = *(_QWORD **)(a1 + 120);
    v33 = 4 * v31;
    v34 = *(unsigned int *)(a1 + 136);
    v35 = &v32[2 * v34];
    if ( (unsigned int)(4 * v31) < 0x40 )
      v33 = 64;
    if ( v33 >= (unsigned int)v34 )
    {
      while ( v35 != v32 )
      {
        if ( *v32 != -8 )
        {
          if ( *v32 != -16 )
          {
            v36 = v32[1];
            if ( v36 )
            {
              _libc_free(*(_QWORD *)(v36 + 48));
              _libc_free(*(_QWORD *)(v36 + 24));
              j_j___libc_free_0(v36, 72);
            }
          }
          *v32 = -8;
        }
        v32 += 2;
      }
      goto LABEL_37;
    }
    do
    {
      if ( *v32 != -16 && *v32 != -8 )
      {
        v59 = v32[1];
        if ( v59 )
        {
          _libc_free(*(_QWORD *)(v59 + 48));
          _libc_free(*(_QWORD *)(v59 + 24));
          j_j___libc_free_0(v59, 72);
        }
      }
      v32 += 2;
    }
    while ( v35 != v32 );
    v61 = *(_DWORD *)(a1 + 136);
    if ( !v31 )
    {
      if ( v61 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 120));
        *(_QWORD *)(a1 + 120) = 0;
        *(_QWORD *)(a1 + 128) = 0;
        *(_DWORD *)(a1 + 136) = 0;
        goto LABEL_38;
      }
LABEL_37:
      *(_QWORD *)(a1 + 128) = 0;
      goto LABEL_38;
    }
    v62 = 64;
    v63 = v31 - 1;
    if ( v63 )
    {
      _BitScanReverse(&v64, v63);
      v62 = 1 << (33 - (v64 ^ 0x1F));
      if ( v62 < 64 )
        v62 = 64;
    }
    v65 = *(_QWORD **)(a1 + 120);
    if ( v61 == v62 )
    {
      *(_QWORD *)(a1 + 128) = 0;
      v73 = &v65[2 * v61];
      do
      {
        if ( v65 )
          *v65 = -8;
        v65 += 2;
      }
      while ( v73 != v65 );
    }
    else
    {
      j___libc_free_0(v65);
      v66 = ((((((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
               | (4 * v62 / 3u + 1)
               | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
             | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
             | (4 * v62 / 3u + 1)
             | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
             | (4 * v62 / 3u + 1)
             | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
           | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
           | (4 * v62 / 3u + 1)
           | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 16;
      v67 = (v66
           | (((((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
               | (4 * v62 / 3u + 1)
               | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
             | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
             | (4 * v62 / 3u + 1)
             | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
             | (4 * v62 / 3u + 1)
             | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
           | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
           | (4 * v62 / 3u + 1)
           | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 136) = v67;
      v68 = (_QWORD *)sub_22077B0(16 * v67);
      v69 = *(unsigned int *)(a1 + 136);
      *(_QWORD *)(a1 + 128) = 0;
      *(_QWORD *)(a1 + 120) = v68;
      for ( k = &v68[2 * v69]; k != v68; v68 += 2 )
      {
        if ( v68 )
          *v68 = -8;
      }
    }
  }
LABEL_38:
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  sub_21EB2D0(a1);
  sub_1DFEBA0((_QWORD *)a1);
  if ( v77 )
    j_j___libc_free_0(v77, v79 - v77);
}
