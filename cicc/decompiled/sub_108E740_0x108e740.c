// Function: sub_108E740
// Address: 0x108e740
//
__int64 __fastcall sub_108E740(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rdx
  _QWORD *v3; // rax
  _QWORD *i; // rdx
  int v5; // eax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *m; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 *v16; // r13
  unsigned __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r13
  __int64 v22; // r14
  _QWORD *v23; // rbx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 *v30; // r15
  unsigned __int64 v31; // r12
  __int64 v32; // rdi
  _QWORD *v33; // r12
  _QWORD *n; // rbx
  __int64 (__fastcall *v35)(__int64); // rax
  _QWORD *v36; // rdi
  _QWORD *v37; // r12
  _QWORD *ii; // rbx
  __int64 v39; // rax
  _QWORD *v40; // rdi
  _QWORD *v41; // r12
  _QWORD *v42; // rdi
  unsigned int v44; // ecx
  unsigned int v45; // eax
  int v46; // ebx
  _QWORD *v47; // rdi
  _QWORD *v48; // rax
  unsigned int v49; // ecx
  unsigned int v50; // eax
  int v51; // ebx
  _QWORD *v52; // rdi
  unsigned int v53; // kr00_4
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rax
  _QWORD *v56; // rax
  __int64 v57; // rdx
  _QWORD *j; // rdx
  unsigned int v59; // kr04_4
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // rdx
  _QWORD *k; // rdx
  _QWORD *v65; // rax
  __int64 v67; // [rsp+18h] [rbp-B8h]
  __int64 *v68; // [rsp+20h] [rbp-B0h]
  __int64 v69; // [rsp+38h] [rbp-98h]
  __int64 v70; // [rsp+40h] [rbp-90h]
  __int64 v71; // [rsp+48h] [rbp-88h]
  __int64 v72; // [rsp+50h] [rbp-80h]
  __int64 v73; // [rsp+58h] [rbp-78h]
  __int64 v74; // [rsp+58h] [rbp-78h]
  __int64 v75; // [rsp+60h] [rbp-70h] BYREF
  __int64 v76; // [rsp+68h] [rbp-68h]
  __int64 v77; // [rsp+70h] [rbp-60h]
  __int64 v78; // [rsp+78h] [rbp-58h]
  __int64 v79; // [rsp+80h] [rbp-50h] BYREF
  __int64 v80; // [rsp+88h] [rbp-48h]
  __int64 v81; // [rsp+90h] [rbp-40h]
  __int64 v82; // [rsp+98h] [rbp-38h]

  v1 = *(_DWORD *)(a1 + 296);
  ++*(_QWORD *)(a1 + 280);
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 300) )
      goto LABEL_7;
    v2 = *(unsigned int *)(a1 + 304);
    if ( (unsigned int)v2 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 288), 16 * v2, 8);
      *(_QWORD *)(a1 + 288) = 0;
      *(_QWORD *)(a1 + 296) = 0;
      *(_DWORD *)(a1 + 304) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v2 = *(unsigned int *)(a1 + 304);
  v49 = 4 * v1;
  if ( (unsigned int)(4 * v1) < 0x40 )
    v49 = 64;
  if ( v49 >= (unsigned int)v2 )
  {
LABEL_4:
    v3 = *(_QWORD **)(a1 + 288);
    for ( i = &v3[2 * v2]; i != v3; v3 += 2 )
      *v3 = -4096;
    *(_QWORD *)(a1 + 296) = 0;
    goto LABEL_7;
  }
  v50 = v1 - 1;
  if ( !v50 )
  {
    v51 = 64;
    v52 = *(_QWORD **)(a1 + 288);
LABEL_57:
    sub_C7D6A0((__int64)v52, 16 * v2, 8);
    v53 = 4 * v51;
    v54 = ((((((((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
             | (v53 / 3 + 1)
             | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 4)
           | (((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
           | (v53 / 3 + 1)
           | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 8)
         | (((((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
           | (v53 / 3 + 1)
           | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 4)
         | (((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
         | (v53 / 3 + 1)
         | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 16;
    v55 = (v54
         | (((((((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
             | (v53 / 3 + 1)
             | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 4)
           | (((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
           | (v53 / 3 + 1)
           | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 8)
         | (((((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
           | (v53 / 3 + 1)
           | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 4)
         | (((v53 / 3 + 1) | ((unsigned __int64)(v53 / 3 + 1) >> 1)) >> 2)
         | (v53 / 3 + 1)
         | ((unsigned __int64)(v53 / 3 + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 304) = v55;
    v56 = (_QWORD *)sub_C7D670(16 * v55, 8);
    v57 = *(unsigned int *)(a1 + 304);
    *(_QWORD *)(a1 + 296) = 0;
    *(_QWORD *)(a1 + 288) = v56;
    for ( j = &v56[2 * v57]; j != v56; v56 += 2 )
    {
      if ( v56 )
        *v56 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v50, v50);
  v51 = 1 << (33 - (v50 ^ 0x1F));
  v52 = *(_QWORD **)(a1 + 288);
  if ( v51 < 64 )
    v51 = 64;
  if ( v51 != (_DWORD)v2 )
    goto LABEL_57;
  *(_QWORD *)(a1 + 296) = 0;
  v65 = &v52[2 * (unsigned int)v51];
  do
  {
    if ( v52 )
      *v52 = -4096;
    v52 += 2;
  }
  while ( v65 != v52 );
LABEL_7:
  v5 = *(_DWORD *)(a1 + 264);
  ++*(_QWORD *)(a1 + 248);
  if ( v5 )
  {
    v6 = *(unsigned int *)(a1 + 272);
    v44 = 4 * v5;
    if ( (unsigned int)(4 * v5) < 0x40 )
      v44 = 64;
    if ( v44 < (unsigned int)v6 )
    {
      v45 = v5 - 1;
      if ( v45 )
      {
        _BitScanReverse(&v45, v45);
        v46 = 1 << (33 - (v45 ^ 0x1F));
        v47 = *(_QWORD **)(a1 + 256);
        if ( v46 < 64 )
          v46 = 64;
        if ( v46 == (_DWORD)v6 )
        {
          *(_QWORD *)(a1 + 264) = 0;
          v48 = &v47[2 * (unsigned int)v46];
          do
          {
            if ( v47 )
              *v47 = -4096;
            v47 += 2;
          }
          while ( v48 != v47 );
          goto LABEL_13;
        }
      }
      else
      {
        v46 = 64;
        v47 = *(_QWORD **)(a1 + 256);
      }
      sub_C7D6A0((__int64)v47, 16 * v6, 8);
      v59 = 4 * v46;
      v60 = ((((((((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
               | (v59 / 3 + 1)
               | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 4)
             | (((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
             | (v59 / 3 + 1)
             | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 8)
           | (((((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
             | (v59 / 3 + 1)
             | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 4)
           | (((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
           | (v59 / 3 + 1)
           | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 16;
      v61 = (v60
           | (((((((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
               | (v59 / 3 + 1)
               | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 4)
             | (((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
             | (v59 / 3 + 1)
             | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 8)
           | (((((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
             | (v59 / 3 + 1)
             | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 4)
           | (((v59 / 3 + 1) | ((unsigned __int64)(v59 / 3 + 1) >> 1)) >> 2)
           | (v59 / 3 + 1)
           | ((unsigned __int64)(v59 / 3 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 272) = v61;
      v62 = (_QWORD *)sub_C7D670(16 * v61, 8);
      v63 = *(unsigned int *)(a1 + 272);
      *(_QWORD *)(a1 + 264) = 0;
      *(_QWORD *)(a1 + 256) = v62;
      for ( k = &v62[2 * v63]; k != v62; v62 += 2 )
      {
        if ( v62 )
          *v62 = -4096;
      }
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  if ( *(_DWORD *)(a1 + 268) )
  {
    v6 = *(unsigned int *)(a1 + 272);
    if ( (unsigned int)v6 <= 0x40 )
    {
LABEL_10:
      v7 = *(_QWORD **)(a1 + 256);
      for ( m = &v7[2 * v6]; m != v7; v7 += 2 )
        *v7 = -4096;
      *(_QWORD *)(a1 + 264) = 0;
      goto LABEL_13;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 256), 16 * v6, 8);
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 264) = 0;
    *(_DWORD *)(a1 + 272) = 0;
  }
LABEL_13:
  v9 = *(_QWORD *)(a1 + 328);
  v10 = *(_QWORD *)(a1 + 360);
  v11 = *(_QWORD *)(a1 + 352);
  v12 = *(_QWORD *)(a1 + 336);
  v13 = *(_QWORD *)(a1 + 344);
  v82 = *(_QWORD *)(a1 + 384);
  v14 = *(_QWORD *)(a1 + 368);
  v15 = *(_QWORD *)(a1 + 376);
  v73 = v9;
  v16 = (__int64 *)(v11 + 8);
  v79 = v10;
  v75 = v9;
  v80 = v14;
  v81 = v15;
  v76 = v12;
  v77 = v13;
  v78 = v11;
  sub_108B970(&v75, &v79);
  v17 = *(_QWORD *)(a1 + 384) + 8LL;
  if ( v17 > v11 + 8 )
  {
    do
    {
      v18 = *v16++;
      j_j___libc_free_0(v18, 480);
    }
    while ( v17 > (unsigned __int64)v16 );
  }
  *(_QWORD *)(a1 + 368) = v12;
  *(_QWORD *)(a1 + 360) = v73;
  *(_QWORD *)(a1 + 376) = v13;
  *(_QWORD *)(a1 + 384) = v11;
  v67 = a1 + 1752;
  do
  {
    v19 = *(_QWORD *)v67;
    v20 = *(_QWORD *)(*(_QWORD *)v67 + 96LL);
    *(_QWORD *)(v19 + 16) = 0;
    *(_QWORD *)(v19 + 24) = 0;
    v21 = *(_QWORD *)(v19 + 80);
    *(_QWORD *)(v19 + 32) = 0;
    *(_QWORD *)(v19 + 40) = 0;
    *(_DWORD *)(v19 + 48) = 0;
    *(_WORD *)(v19 + 56) = -3;
    v70 = v20;
    v69 = *(_QWORD *)(v19 + 112);
    v68 = (__int64 *)(*(_QWORD *)(v19 + 104) + 8LL);
LABEL_17:
    v22 = v21;
    while ( v69 != v22 )
    {
      v23 = *(_QWORD **)v22;
      v24 = *(_QWORD *)(*(_QWORD *)v22 + 24LL);
      v25 = *(_QWORD *)(*(_QWORD *)v22 + 32LL);
      v26 = *(_QWORD *)(*(_QWORD *)v22 + 40LL);
      v27 = *(_QWORD *)(*(_QWORD *)v22 + 72LL);
      v74 = *(_QWORD *)(*(_QWORD *)v22 + 16LL);
      v28 = *(_QWORD *)(*(_QWORD *)v22 + 56LL);
      v29 = *(_QWORD *)(*(_QWORD *)v22 + 64LL);
      v79 = *(_QWORD *)(*(_QWORD *)v22 + 48LL);
      v75 = v74;
      v30 = (__int64 *)(v26 + 8);
      v82 = v27;
      v72 = v24;
      v71 = v25;
      v80 = v28;
      v81 = v29;
      v76 = v24;
      v77 = v25;
      v78 = v26;
      sub_108B970(&v75, &v79);
      v31 = v23[9] + 8LL;
      if ( v31 > v26 + 8 )
      {
        do
        {
          v32 = *v30++;
          j_j___libc_free_0(v32, 480);
        }
        while ( v31 > (unsigned __int64)v30 );
      }
      v23[9] = v26;
      v22 += 8;
      v23[6] = v74;
      v23[7] = v72;
      v23[8] = v71;
      if ( v70 == v22 )
      {
        v21 = *v68++;
        v70 = v21 + 512;
        goto LABEL_17;
      }
    }
    v67 += 8;
  }
  while ( a1 + 1792 != v67 );
  v33 = *(_QWORD **)(a1 + 1800);
  for ( n = *(_QWORD **)(a1 + 1792); v33 != n; n += 10 )
  {
    while ( 1 )
    {
      v35 = *(__int64 (__fastcall **)(__int64))(*n + 8LL);
      if ( v35 != sub_108ADC0 )
        break;
      n[2] = 0;
      n += 10;
      *(n - 7) = 0;
      *(n - 6) = 0;
      *(n - 5) = 0;
      *((_DWORD *)n - 8) = 0;
      *((_WORD *)n - 12) = -3;
      if ( v33 == n )
        goto LABEL_29;
    }
    v36 = n;
    v35((__int64)v36);
  }
LABEL_29:
  v37 = *(_QWORD **)(a1 + 1824);
  for ( ii = *(_QWORD **)(a1 + 1816); v37 != ii; ii += 8 )
  {
    v39 = *ii;
    v40 = ii;
    (*(void (__fastcall **)(_QWORD *))(v39 + 8))(v40);
  }
  v41 = *(_QWORD **)(a1 + 2024);
  *(_QWORD *)(a1 + 1856) = 0;
  *(_QWORD *)(a1 + 1864) = 0;
  *(_QWORD *)(a1 + 1872) = 0;
  *(_QWORD *)(a1 + 1880) = 0;
  *(_DWORD *)(a1 + 1888) = 0;
  *(_WORD *)(a1 + 1896) = -3;
  *(_QWORD *)(a1 + 1976) = 0;
  *(_QWORD *)(a1 + 1984) = 0;
  *(_QWORD *)(a1 + 1992) = 0;
  *(_QWORD *)(a1 + 2000) = 0;
  *(_DWORD *)(a1 + 2008) = 0;
  *(_WORD *)(a1 + 2016) = -3;
  *(_QWORD *)(a1 + 2024) = 0;
  if ( v41 )
  {
    v42 = (_QWORD *)v41[4];
    if ( v42 != v41 + 6 )
      j_j___libc_free_0(v42, v41[6] + 1LL);
    if ( (_QWORD *)*v41 != v41 + 2 )
      j_j___libc_free_0(*v41, v41[2] + 1LL);
    j_j___libc_free_0(v41, 72);
  }
  *(_WORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 156) = 0;
  sub_C0C1A0(a1 + 192);
  return sub_E8EB90(a1);
}
