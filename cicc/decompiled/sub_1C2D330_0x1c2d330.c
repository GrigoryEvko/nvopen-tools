// Function: sub_1C2D330
// Address: 0x1c2d330
//
void __fastcall sub_1C2D330(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  int v4; // r14d
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // r11
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r10
  unsigned int v20; // esi
  int v21; // ecx
  int v22; // ecx
  __int64 v23; // r8
  unsigned int v24; // esi
  int v25; // edx
  __int64 v26; // rdi
  int v27; // eax
  __int64 v28; // rdx
  _QWORD *v29; // rax
  _QWORD *i; // rdx
  __int64 v31; // rax
  int v32; // r14d
  _QWORD *v33; // rbx
  unsigned int v34; // eax
  __int64 v35; // rdx
  _QWORD *v36; // r13
  __int64 v37; // r14
  int v38; // edx
  __int64 *v39; // r10
  int v40; // edi
  int v41; // ecx
  int v42; // ecx
  __int64 *v43; // r8
  unsigned int v44; // r15d
  __int64 v45; // rdi
  int v46; // r10d
  __int64 v47; // rsi
  unsigned int v48; // ecx
  _QWORD *v49; // rdi
  unsigned int v50; // eax
  int v51; // eax
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rax
  int v54; // ebx
  __int64 v55; // r13
  _QWORD *v56; // rax
  __int64 v57; // rdx
  _QWORD *j; // rdx
  __int64 v59; // r15
  int v60; // ebx
  int v61; // edx
  int v62; // ebx
  unsigned int v63; // r14d
  unsigned int v64; // eax
  _QWORD *v65; // rdi
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rdi
  _QWORD *v68; // rax
  __int64 v69; // rdx
  _QWORD *k; // rdx
  int v71; // r15d
  __int64 *v72; // r10
  _QWORD *v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // [rsp+0h] [rbp-60h]
  __int64 v76; // [rsp+0h] [rbp-60h]
  __int64 v77; // [rsp+8h] [rbp-58h]
  int v78; // [rsp+8h] [rbp-58h]
  __int64 v79; // [rsp+8h] [rbp-58h]
  __int64 v80; // [rsp+10h] [rbp-50h] BYREF
  __int64 v81; // [rsp+18h] [rbp-48h]
  __int64 v82; // [rsp+20h] [rbp-40h]

  v2 = *(_QWORD *)a1;
  v80 = 0;
  v81 = 0;
  v3 = *(_QWORD *)(v2 + 80);
  v82 = 0;
  if ( v3 )
    v3 -= 24;
  v4 = 0;
  sub_191E690((__int64)&v80, v3);
  v5 = v80;
  v6 = v81;
  v7 = a1 + 144;
  if ( v80 != v81 )
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(a1 + 16);
      v13 = 0;
      v14 = *(unsigned int *)(v12 + 48);
      if ( !(_DWORD)v14 )
        goto LABEL_11;
      v15 = *(_QWORD *)(v6 - 8);
      v16 = *(_QWORD *)(v12 + 32);
      v17 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v15 == *v18 )
      {
LABEL_9:
        if ( v18 != (__int64 *)(v16 + 16 * v14) )
        {
          v13 = v18[1];
          goto LABEL_11;
        }
      }
      else
      {
        v38 = 1;
        while ( v19 != -8 )
        {
          v60 = v38 + 1;
          v17 = (v14 - 1) & (v38 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v15 == *v18 )
            goto LABEL_9;
          v38 = v60;
        }
      }
      v13 = 0;
LABEL_11:
      v20 = *(_DWORD *)(a1 + 168);
      ++v4;
      if ( !v20 )
      {
        ++*(_QWORD *)(a1 + 144);
        goto LABEL_13;
      }
      v8 = *(_QWORD *)(a1 + 152);
      v9 = (v20 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v13 == *v10 )
      {
LABEL_6:
        v6 -= 8;
        *((_DWORD *)v10 + 2) = v4;
        if ( v5 == v6 )
          break;
      }
      else
      {
        v78 = 1;
        v39 = 0;
        while ( v11 != -8 )
        {
          if ( v11 == -16 && !v39 )
            v39 = v10;
          v9 = (v20 - 1) & (v78 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v13 == *v10 )
            goto LABEL_6;
          ++v78;
        }
        v40 = *(_DWORD *)(a1 + 160);
        if ( v39 )
          v10 = v39;
        ++*(_QWORD *)(a1 + 144);
        v25 = v40 + 1;
        if ( 4 * (v40 + 1) < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a1 + 164) - v25 <= v20 >> 3 )
          {
            v76 = v5;
            v79 = v7;
            sub_19F5120(v7, v20);
            v41 = *(_DWORD *)(a1 + 168);
            if ( !v41 )
            {
LABEL_123:
              ++*(_DWORD *)(a1 + 160);
              BUG();
            }
            v42 = v41 - 1;
            v43 = 0;
            v7 = v79;
            v5 = v76;
            v44 = v42 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v45 = *(_QWORD *)(a1 + 152);
            v46 = 1;
            v25 = *(_DWORD *)(a1 + 160) + 1;
            v10 = (__int64 *)(v45 + 16LL * v44);
            v47 = *v10;
            if ( v13 != *v10 )
            {
              while ( v47 != -8 )
              {
                if ( !v43 && v47 == -16 )
                  v43 = v10;
                v44 = v42 & (v46 + v44);
                v10 = (__int64 *)(v45 + 16LL * v44);
                v47 = *v10;
                if ( v13 == *v10 )
                  goto LABEL_15;
                ++v46;
              }
              if ( v43 )
                v10 = v43;
            }
          }
          goto LABEL_15;
        }
LABEL_13:
        v75 = v5;
        v77 = v7;
        sub_19F5120(v7, 2 * v20);
        v21 = *(_DWORD *)(a1 + 168);
        if ( !v21 )
          goto LABEL_123;
        v22 = v21 - 1;
        v23 = *(_QWORD *)(a1 + 152);
        v7 = v77;
        v5 = v75;
        v24 = v22 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v25 = *(_DWORD *)(a1 + 160) + 1;
        v10 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v10;
        if ( v13 != *v10 )
        {
          v71 = 1;
          v72 = 0;
          while ( v26 != -8 )
          {
            if ( v26 == -16 && !v72 )
              v72 = v10;
            v24 = v22 & (v71 + v24);
            v10 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v10;
            if ( v13 == *v10 )
              goto LABEL_15;
            ++v71;
          }
          if ( v72 )
            v10 = v72;
        }
LABEL_15:
        *(_DWORD *)(a1 + 160) = v25;
        if ( *v10 != -8 )
          --*(_DWORD *)(a1 + 164);
        v6 -= 8;
        *((_DWORD *)v10 + 2) = 0;
        *v10 = v13;
        *((_DWORD *)v10 + 2) = v4;
        if ( v5 == v6 )
          break;
      }
    }
  }
  sub_3952FD0(a1);
  v27 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  if ( !v27 )
  {
    if ( !*(_DWORD *)(a1 + 76) )
      goto LABEL_24;
    v28 = *(unsigned int *)(a1 + 80);
    if ( (unsigned int)v28 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 64));
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 80) = 0;
      goto LABEL_24;
    }
    goto LABEL_21;
  }
  v48 = 4 * v27;
  v28 = *(unsigned int *)(a1 + 80);
  if ( (unsigned int)(4 * v27) < 0x40 )
    v48 = 64;
  if ( v48 >= (unsigned int)v28 )
  {
LABEL_21:
    v29 = *(_QWORD **)(a1 + 64);
    for ( i = &v29[2 * v28]; i != v29; v29 += 2 )
      *v29 = -8;
    *(_QWORD *)(a1 + 72) = 0;
    goto LABEL_24;
  }
  v49 = *(_QWORD **)(a1 + 64);
  v50 = v27 - 1;
  if ( !v50 )
  {
    v55 = 2048;
    v54 = 128;
LABEL_66:
    j___libc_free_0(v49);
    *(_DWORD *)(a1 + 80) = v54;
    v56 = (_QWORD *)sub_22077B0(v55);
    v57 = *(unsigned int *)(a1 + 80);
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 64) = v56;
    for ( j = &v56[2 * v57]; j != v56; v56 += 2 )
    {
      if ( v56 )
        *v56 = -8;
    }
    goto LABEL_24;
  }
  _BitScanReverse(&v50, v50);
  v51 = 1 << (33 - (v50 ^ 0x1F));
  if ( v51 < 64 )
    v51 = 64;
  if ( (_DWORD)v28 != v51 )
  {
    v52 = (((4 * v51 / 3u + 1) | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)) >> 2)
        | (4 * v51 / 3u + 1)
        | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)
        | (((((4 * v51 / 3u + 1) | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)) >> 2)
          | (4 * v51 / 3u + 1)
          | ((unsigned __int64)(4 * v51 / 3u + 1) >> 1)) >> 4);
    v53 = (v52 >> 8) | v52;
    v54 = (v53 | (v53 >> 16)) + 1;
    v55 = 16 * ((v53 | (v53 >> 16)) + 1);
    goto LABEL_66;
  }
  *(_QWORD *)(a1 + 72) = 0;
  v73 = &v49[2 * (unsigned int)v28];
  do
  {
    if ( v49 )
      *v49 = -8;
    v49 += 2;
  }
  while ( v73 != v49 );
LABEL_24:
  v31 = *(_QWORD *)(a1 + 88);
  if ( v31 != *(_QWORD *)(a1 + 96) )
    *(_QWORD *)(a1 + 96) = v31;
  v32 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v32 || *(_DWORD *)(a1 + 132) )
  {
    v33 = *(_QWORD **)(a1 + 120);
    v34 = 4 * v32;
    v35 = *(unsigned int *)(a1 + 136);
    v36 = &v33[2 * v35];
    if ( (unsigned int)(4 * v32) < 0x40 )
      v34 = 64;
    if ( v34 >= (unsigned int)v35 )
    {
      while ( v33 != v36 )
      {
        if ( *v33 != -8 )
        {
          if ( *v33 != -16 )
          {
            v37 = v33[1];
            if ( v37 )
            {
              _libc_free(*(_QWORD *)(v37 + 48));
              _libc_free(*(_QWORD *)(v37 + 24));
              j_j___libc_free_0(v37, 72);
            }
          }
          *v33 = -8;
        }
        v33 += 2;
      }
      goto LABEL_39;
    }
    do
    {
      if ( *v33 != -16 && *v33 != -8 )
      {
        v59 = v33[1];
        if ( v59 )
        {
          _libc_free(*(_QWORD *)(v59 + 48));
          _libc_free(*(_QWORD *)(v59 + 24));
          j_j___libc_free_0(v59, 72);
        }
      }
      v33 += 2;
    }
    while ( v33 != v36 );
    v61 = *(_DWORD *)(a1 + 136);
    if ( !v32 )
    {
      if ( v61 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 120));
        *(_QWORD *)(a1 + 120) = 0;
        *(_QWORD *)(a1 + 128) = 0;
        *(_DWORD *)(a1 + 136) = 0;
        goto LABEL_40;
      }
LABEL_39:
      *(_QWORD *)(a1 + 128) = 0;
      goto LABEL_40;
    }
    v62 = 64;
    v63 = v32 - 1;
    if ( v63 )
    {
      _BitScanReverse(&v64, v63);
      v62 = 1 << (33 - (v64 ^ 0x1F));
      if ( v62 < 64 )
        v62 = 64;
    }
    v65 = *(_QWORD **)(a1 + 120);
    if ( v62 == v61 )
    {
      *(_QWORD *)(a1 + 128) = 0;
      v74 = &v65[2 * (unsigned int)v62];
      do
      {
        if ( v65 )
          *v65 = -8;
        v65 += 2;
      }
      while ( v74 != v65 );
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
LABEL_40:
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  sub_3954A10(a1);
  sub_1C2CBE0((__int64 *)a1);
  if ( v80 )
    j_j___libc_free_0(v80, v82 - v80);
}
