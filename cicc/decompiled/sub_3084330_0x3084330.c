// Function: sub_3084330
// Address: 0x3084330
//
void __fastcall sub_3084330(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r14d
  __int64 v7; // r12
  __int64 v8; // rsi
  unsigned __int64 v9; // r8
  __int64 v10; // r11
  _BYTE *v11; // r13
  __int64 v12; // r9
  unsigned int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // eax
  __int64 v20; // rbx
  unsigned int v21; // esi
  int v22; // esi
  int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // ecx
  int v26; // eax
  __int64 *v27; // rdx
  __int64 v28; // rdi
  int v29; // eax
  __int64 v30; // rdx
  _DWORD *v31; // rax
  _DWORD *i; // rdx
  __int64 v33; // rax
  int v34; // r15d
  _QWORD *v35; // rbx
  unsigned int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // r14
  _QWORD *v39; // r13
  unsigned __int64 v40; // r14
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  int v43; // eax
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // rdi
  __int64 *v47; // r9
  unsigned int v48; // r15d
  int v49; // r10d
  __int64 v50; // rsi
  unsigned int v51; // ecx
  unsigned int v52; // eax
  _DWORD *v53; // rdi
  int v54; // ebx
  _DWORD *v55; // rax
  _QWORD *v56; // r12
  unsigned __int64 v57; // r13
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rdi
  int v60; // edx
  int v61; // ebx
  unsigned int v62; // eax
  _QWORD *v63; // rdi
  unsigned __int64 v64; // rax
  unsigned __int64 v65; // rdi
  _QWORD *v66; // rax
  __int64 v67; // rdx
  _QWORD *k; // rdx
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rax
  _DWORD *v71; // rax
  __int64 v72; // rdx
  _DWORD *j; // rdx
  int v74; // r15d
  __int64 *v75; // r10
  _QWORD *v76; // rax
  unsigned __int64 v77; // [rsp+8h] [rbp-98h]
  unsigned __int64 v78; // [rsp+8h] [rbp-98h]
  __int64 v79; // [rsp+10h] [rbp-90h]
  int v80; // [rsp+10h] [rbp-90h]
  __int64 v81; // [rsp+10h] [rbp-90h]
  __int64 v82; // [rsp+10h] [rbp-90h]
  _BYTE *v83; // [rsp+20h] [rbp-80h] BYREF
  __int64 v84; // [rsp+28h] [rbp-78h]
  _BYTE v85[112]; // [rsp+30h] [rbp-70h] BYREF

  v6 = 0;
  v7 = (__int64)a1;
  v8 = *a1;
  v83 = v85;
  v84 = 0x800000000LL;
  sub_3083F80((__int64)&v83, v8, a3, a4, a5, a6);
  v9 = (unsigned __int64)v83;
  v10 = (__int64)(a1 + 18);
  v11 = &v83[8 * (unsigned int)v84];
  if ( v83 != v11 )
  {
    while ( 1 )
    {
      v16 = *((_QWORD *)v11 - 1);
      v17 = *(_QWORD *)(v7 + 16);
      if ( v16 )
      {
        v18 = (unsigned int)(*(_DWORD *)(v16 + 24) + 1);
        v19 = *(_DWORD *)(v16 + 24) + 1;
      }
      else
      {
        v18 = 0;
        v19 = 0;
      }
      v20 = 0;
      if ( v19 < *(_DWORD *)(v17 + 32) )
        v20 = *(_QWORD *)(*(_QWORD *)(v17 + 24) + 8 * v18);
      v21 = *(_DWORD *)(v7 + 168);
      ++v6;
      if ( !v21 )
        break;
      v12 = *(_QWORD *)(v7 + 152);
      v13 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v20 == *v14 )
      {
LABEL_4:
        v11 -= 8;
        *((_DWORD *)v14 + 2) = v6;
        if ( (_BYTE *)v9 == v11 )
          goto LABEL_16;
      }
      else
      {
        v80 = 1;
        v27 = 0;
        while ( v15 != -4096 )
        {
          if ( v15 == -8192 && !v27 )
            v27 = v14;
          v13 = (v21 - 1) & (v80 + v13);
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v20 == *v14 )
            goto LABEL_4;
          ++v80;
        }
        if ( !v27 )
          v27 = v14;
        v43 = *(_DWORD *)(v7 + 160);
        ++*(_QWORD *)(v7 + 144);
        v26 = v43 + 1;
        if ( 4 * v26 < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(v7 + 164) - v26 <= v21 >> 3 )
          {
            v78 = v9;
            v81 = v10;
            sub_3083DA0(v10, v21);
            v44 = *(_DWORD *)(v7 + 168);
            if ( !v44 )
            {
LABEL_123:
              ++*(_DWORD *)(v7 + 160);
              BUG();
            }
            v45 = v44 - 1;
            v46 = *(_QWORD *)(v7 + 152);
            v47 = 0;
            v48 = v45 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
            v10 = v81;
            v9 = v78;
            v49 = 1;
            v26 = *(_DWORD *)(v7 + 160) + 1;
            v27 = (__int64 *)(v46 + 16LL * v48);
            v50 = *v27;
            if ( v20 != *v27 )
            {
              while ( v50 != -4096 )
              {
                if ( v50 == -8192 && !v47 )
                  v47 = v27;
                v48 = v45 & (v49 + v48);
                v27 = (__int64 *)(v46 + 16LL * v48);
                v50 = *v27;
                if ( v20 == *v27 )
                  goto LABEL_13;
                ++v49;
              }
              if ( v47 )
                v27 = v47;
            }
          }
          goto LABEL_13;
        }
LABEL_11:
        v77 = v9;
        v79 = v10;
        sub_3083DA0(v10, 2 * v21);
        v22 = *(_DWORD *)(v7 + 168);
        if ( !v22 )
          goto LABEL_123;
        v23 = v22 - 1;
        v24 = *(_QWORD *)(v7 + 152);
        v10 = v79;
        v9 = v77;
        v25 = v23 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v26 = *(_DWORD *)(v7 + 160) + 1;
        v27 = (__int64 *)(v24 + 16LL * v25);
        v28 = *v27;
        if ( v20 != *v27 )
        {
          v74 = 1;
          v75 = 0;
          while ( v28 != -4096 )
          {
            if ( v28 == -8192 && !v75 )
              v75 = v27;
            v25 = v23 & (v74 + v25);
            v27 = (__int64 *)(v24 + 16LL * v25);
            v28 = *v27;
            if ( v20 == *v27 )
              goto LABEL_13;
            ++v74;
          }
          if ( v75 )
            v27 = v75;
        }
LABEL_13:
        *(_DWORD *)(v7 + 160) = v26;
        if ( *v27 != -4096 )
          --*(_DWORD *)(v7 + 164);
        v11 -= 8;
        *v27 = v20;
        *((_DWORD *)v27 + 2) = 0;
        *((_DWORD *)v27 + 2) = v6;
        if ( (_BYTE *)v9 == v11 )
          goto LABEL_16;
      }
    }
    ++*(_QWORD *)(v7 + 144);
    goto LABEL_11;
  }
LABEL_16:
  v29 = *(_DWORD *)(v7 + 72);
  ++*(_QWORD *)(v7 + 56);
  if ( !v29 )
  {
    if ( !*(_DWORD *)(v7 + 76) )
      goto LABEL_22;
    v30 = *(unsigned int *)(v7 + 80);
    if ( (unsigned int)v30 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(v7 + 64), 8 * v30, 4);
      *(_QWORD *)(v7 + 64) = 0;
      *(_QWORD *)(v7 + 72) = 0;
      *(_DWORD *)(v7 + 80) = 0;
      goto LABEL_22;
    }
    goto LABEL_19;
  }
  v51 = 4 * v29;
  v30 = *(unsigned int *)(v7 + 80);
  if ( (unsigned int)(4 * v29) < 0x40 )
    v51 = 64;
  if ( v51 >= (unsigned int)v30 )
  {
LABEL_19:
    v31 = *(_DWORD **)(v7 + 64);
    for ( i = &v31[2 * v30]; i != v31; v31 += 2 )
      *v31 = -1;
    *(_QWORD *)(v7 + 72) = 0;
    goto LABEL_22;
  }
  v52 = v29 - 1;
  if ( v52 )
  {
    _BitScanReverse(&v52, v52);
    v53 = *(_DWORD **)(v7 + 64);
    v54 = 1 << (33 - (v52 ^ 0x1F));
    if ( v54 < 64 )
      v54 = 64;
    if ( (_DWORD)v30 == v54 )
    {
      *(_QWORD *)(v7 + 72) = 0;
      v55 = &v53[2 * v30];
      do
      {
        if ( v53 )
          *v53 = -1;
        v53 += 2;
      }
      while ( v55 != v53 );
      goto LABEL_22;
    }
  }
  else
  {
    v53 = *(_DWORD **)(v7 + 64);
    v54 = 64;
  }
  sub_C7D6A0((__int64)v53, 8 * v30, 4);
  v69 = ((((((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
           | (4 * v54 / 3u + 1)
           | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
         | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
         | (4 * v54 / 3u + 1)
         | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
         | (4 * v54 / 3u + 1)
         | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
       | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
       | (4 * v54 / 3u + 1)
       | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 16;
  v70 = (v69
       | (((((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
           | (4 * v54 / 3u + 1)
           | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
         | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
         | (4 * v54 / 3u + 1)
         | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
         | (4 * v54 / 3u + 1)
         | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
       | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
       | (4 * v54 / 3u + 1)
       | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(v7 + 80) = v70;
  v71 = (_DWORD *)sub_C7D670(8 * v70, 4);
  v72 = *(unsigned int *)(v7 + 80);
  *(_QWORD *)(v7 + 72) = 0;
  *(_QWORD *)(v7 + 64) = v71;
  for ( j = &v71[2 * v72]; j != v71; v71 += 2 )
  {
    if ( v71 )
      *v71 = -1;
  }
LABEL_22:
  v33 = *(_QWORD *)(v7 + 88);
  if ( v33 != *(_QWORD *)(v7 + 96) )
    *(_QWORD *)(v7 + 96) = v33;
  v34 = *(_DWORD *)(v7 + 128);
  ++*(_QWORD *)(v7 + 112);
  if ( v34 || *(_DWORD *)(v7 + 132) )
  {
    v35 = *(_QWORD **)(v7 + 120);
    v36 = 4 * v34;
    v37 = *(unsigned int *)(v7 + 136);
    v38 = 16 * v37;
    v39 = &v35[2 * v37];
    if ( (unsigned int)(4 * v34) < 0x40 )
      v36 = 64;
    if ( v36 >= (unsigned int)v37 )
    {
      while ( v35 != v39 )
      {
        if ( *v35 != -4096 )
        {
          if ( *v35 != -8192 )
          {
            v40 = v35[1];
            if ( v40 )
            {
              v41 = *(_QWORD *)(v40 + 96);
              if ( v41 != v40 + 112 )
                _libc_free(v41);
              v42 = *(_QWORD *)(v40 + 24);
              if ( v42 != v40 + 40 )
                _libc_free(v42);
              j_j___libc_free_0(v40);
            }
          }
          *v35 = -4096;
        }
        v35 += 2;
      }
    }
    else
    {
      v82 = v7;
      v56 = &v35[2 * v37];
      do
      {
        if ( *v35 != -8192 && *v35 != -4096 )
        {
          v57 = v35[1];
          if ( v57 )
          {
            v58 = *(_QWORD *)(v57 + 96);
            if ( v58 != v57 + 112 )
              _libc_free(v58);
            v59 = *(_QWORD *)(v57 + 24);
            if ( v59 != v57 + 40 )
              _libc_free(v59);
            j_j___libc_free_0(v57);
          }
        }
        v35 += 2;
      }
      while ( v35 != v56 );
      v7 = v82;
      v60 = *(_DWORD *)(v82 + 136);
      if ( v34 )
      {
        v61 = 64;
        if ( v34 != 1 )
        {
          _BitScanReverse(&v62, v34 - 1);
          v61 = 1 << (33 - (v62 ^ 0x1F));
          if ( v61 < 64 )
            v61 = 64;
        }
        v63 = *(_QWORD **)(v82 + 120);
        if ( v61 == v60 )
        {
          *(_QWORD *)(v82 + 128) = 0;
          v76 = &v63[2 * (unsigned int)v61];
          do
          {
            if ( v63 )
              *v63 = -4096;
            v63 += 2;
          }
          while ( v76 != v63 );
        }
        else
        {
          sub_C7D6A0((__int64)v63, v38, 8);
          v64 = ((((((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
                   | (4 * v61 / 3u + 1)
                   | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
                 | (4 * v61 / 3u + 1)
                 | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
                 | (4 * v61 / 3u + 1)
                 | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
               | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
               | (4 * v61 / 3u + 1)
               | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 16;
          v65 = (v64
               | (((((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
                   | (4 * v61 / 3u + 1)
                   | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
                 | (4 * v61 / 3u + 1)
                 | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
                 | (4 * v61 / 3u + 1)
                 | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
               | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
               | (4 * v61 / 3u + 1)
               | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(v82 + 136) = v65;
          v66 = (_QWORD *)sub_C7D670(16 * v65, 8);
          v67 = *(unsigned int *)(v82 + 136);
          *(_QWORD *)(v82 + 128) = 0;
          *(_QWORD *)(v82 + 120) = v66;
          for ( k = &v66[2 * v67]; k != v66; v66 += 2 )
          {
            if ( v66 )
              *v66 = -4096;
          }
        }
        goto LABEL_42;
      }
      if ( v60 )
      {
        sub_C7D6A0(*(_QWORD *)(v82 + 120), v38, 8);
        *(_QWORD *)(v82 + 120) = 0;
        *(_QWORD *)(v82 + 128) = 0;
        *(_DWORD *)(v82 + 136) = 0;
        goto LABEL_42;
      }
    }
    *(_QWORD *)(v7 + 128) = 0;
  }
LABEL_42:
  *(_QWORD *)(v7 + 24) = 0;
  *(_QWORD *)(v7 + 32) = 0;
  sub_307CD20(v7);
  sub_2E592C0((__int64 *)v7);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
}
