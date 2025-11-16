// Function: sub_2BD7690
// Address: 0x2bd7690
//
__int64 __fastcall sub_2BD7690(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r15
  __int64 v9; // r11
  __int64 v10; // rbx
  int v11; // r15d
  __int64 v12; // r14
  unsigned int v13; // eax
  unsigned __int8 *v14; // rsi
  int v15; // edx
  unsigned __int8 *v16; // r12
  __int64 v17; // rcx
  unsigned __int8 v18; // al
  int v19; // eax
  int v20; // edx
  unsigned int v21; // eax
  unsigned __int8 *v22; // rsi
  int v23; // r8d
  int v24; // eax
  __int64 v25; // rcx
  int v26; // eax
  int v27; // edx
  unsigned int v28; // eax
  unsigned __int8 *v29; // rsi
  int v30; // r8d
  unsigned __int8 v31; // al
  int v32; // r14d
  _BYTE *v33; // rsi
  __int64 v34; // rdx
  int v35; // eax
  unsigned int v36; // r14d
  int v37; // eax
  __int64 v38; // rdx
  _QWORD *v39; // rax
  _QWORD *i; // rdx
  __int64 v41; // rax
  _QWORD *v42; // rbx
  _QWORD *v43; // r12
  __int64 v44; // rax
  int v46; // eax
  unsigned int v47; // ecx
  unsigned int v48; // eax
  _QWORD *v49; // rdi
  __int64 v50; // rbx
  _QWORD *v51; // rax
  unsigned __int64 v52; // rdx
  unsigned __int64 v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // rdx
  _QWORD *j; // rdx
  _BYTE *v59; // [rsp+30h] [rbp-70h] BYREF
  __int64 v60; // [rsp+38h] [rbp-68h]
  _BYTE v61[96]; // [rsp+40h] [rbp-60h] BYREF

  v7 = a2;
  v9 = *(_QWORD *)(a2 + 32);
  v59 = v61;
  v60 = 0x200000000LL;
  v10 = v9 + 8LL * *(unsigned int *)(a2 + 40);
  if ( v10 != v9 )
  {
    v11 = 0;
    v12 = v9;
    while ( 1 )
    {
      v15 = *(_DWORD *)(a4 + 2000);
      v16 = *(unsigned __int8 **)(v10 - 8);
      v17 = *(_QWORD *)(a4 + 1984);
      if ( v15 )
      {
        v13 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v14 = *(unsigned __int8 **)(v17 + 8LL * v13);
        if ( v16 == v14 )
          goto LABEL_4;
        a7 = 1;
        while ( v14 != (unsigned __int8 *)-4096LL )
        {
          v13 = (v15 - 1) & (a7 + v13);
          v14 = *(unsigned __int8 **)(v17 + 8LL * v13);
          if ( v16 == v14 )
            goto LABEL_4;
          a7 = (unsigned int)(a7 + 1);
        }
      }
      v18 = *v16;
      if ( (unsigned __int8)(*v16 - 82) <= 1u )
        goto LABEL_4;
      if ( v18 == 94 )
      {
        v19 = sub_2BD14F0(a1, *(_QWORD *)(v10 - 8), a5, a3, a4, 1);
        v17 = *(_QWORD *)(a4 + 1984);
        v15 = *(_DWORD *)(a4 + 2000);
        v11 |= v19;
      }
      else if ( v18 == 91 )
      {
        v46 = sub_2BD17B0(a1, *(_QWORD *)(v10 - 8), a5, a3, a4, 1, a7);
        v17 = *(_QWORD *)(a4 + 1984);
        v15 = *(_DWORD *)(a4 + 2000);
        v11 |= v46;
      }
      if ( v15 )
      {
        v20 = v15 - 1;
        v21 = v20 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v22 = *(unsigned __int8 **)(v17 + 8LL * v21);
        if ( v16 == v22 )
          goto LABEL_4;
        v23 = 1;
        while ( v22 != (unsigned __int8 *)-4096LL )
        {
          a7 = (unsigned int)(v23 + 1);
          v21 = v20 & (v23 + v21);
          v22 = *(unsigned __int8 **)(v17 + 8LL * v21);
          if ( v16 == v22 )
            goto LABEL_4;
          ++v23;
        }
      }
      v24 = sub_2BD62F0(a1, 0, v16, a3, a4, (__int64)&v59, a5);
      v25 = *(_QWORD *)(a4 + 1984);
      v11 |= v24;
      v26 = *(_DWORD *)(a4 + 2000);
      if ( !v26 )
        goto LABEL_17;
      v27 = v26 - 1;
      v28 = (v26 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v29 = *(unsigned __int8 **)(v25 + 8LL * v28);
      if ( v16 == v29 )
      {
LABEL_4:
        v10 -= 8;
        if ( v12 == v10 )
          goto LABEL_20;
      }
      else
      {
        v30 = 1;
        while ( v29 != (unsigned __int8 *)-4096LL )
        {
          a7 = (unsigned int)(v30 + 1);
          v28 = v27 & (v30 + v28);
          v29 = *(unsigned __int8 **)(v25 + 8LL * v28);
          if ( v16 == v29 )
            goto LABEL_4;
          ++v30;
        }
LABEL_17:
        v31 = *v16;
        if ( (unsigned __int8)(*v16 - 82) <= 1u )
          goto LABEL_4;
        if ( v31 != 94 )
        {
          if ( v31 == 91 )
            v11 |= sub_2BD17B0(a1, (__int64)v16, a5, a3, a4, 0, a7);
          goto LABEL_4;
        }
        v10 -= 8;
        v11 |= sub_2BD14F0(a1, (__int64)v16, a5, a3, a4, 0);
        if ( v12 == v10 )
        {
LABEL_20:
          v32 = v11;
          v33 = v59;
          v34 = (unsigned int)v60;
          v7 = a2;
          goto LABEL_21;
        }
      }
    }
  }
  v33 = v61;
  v34 = 0;
  v32 = 0;
LABEL_21:
  v35 = sub_2BCF820(a1, (__int64)v33, v34, a4, a5);
  ++*(_QWORD *)v7;
  v36 = v35 | v32;
  v37 = *(_DWORD *)(v7 + 16);
  if ( !v37 )
  {
    if ( !*(_DWORD *)(v7 + 20) )
      goto LABEL_27;
    v38 = *(unsigned int *)(v7 + 24);
    if ( (unsigned int)v38 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(v7 + 8), 8 * v38, 8);
      *(_QWORD *)(v7 + 8) = 0;
      *(_QWORD *)(v7 + 16) = 0;
      *(_DWORD *)(v7 + 24) = 0;
      goto LABEL_27;
    }
    goto LABEL_24;
  }
  v47 = 4 * v37;
  v38 = *(unsigned int *)(v7 + 24);
  if ( (unsigned int)(4 * v37) < 0x40 )
    v47 = 64;
  if ( (unsigned int)v38 <= v47 )
  {
LABEL_24:
    v39 = *(_QWORD **)(v7 + 8);
    for ( i = &v39[v38]; i != v39; ++v39 )
      *v39 = -4096;
    *(_QWORD *)(v7 + 16) = 0;
    goto LABEL_27;
  }
  v48 = v37 - 1;
  if ( v48 )
  {
    _BitScanReverse(&v48, v48);
    v49 = *(_QWORD **)(v7 + 8);
    v50 = (unsigned int)(1 << (33 - (v48 ^ 0x1F)));
    if ( (int)v50 < 64 )
      v50 = 64;
    if ( (_DWORD)v50 == (_DWORD)v38 )
    {
      *(_QWORD *)(v7 + 16) = 0;
      v51 = &v49[v50];
      do
      {
        if ( v49 )
          *v49 = -4096;
        ++v49;
      }
      while ( v51 != v49 );
      goto LABEL_27;
    }
  }
  else
  {
    v49 = *(_QWORD **)(v7 + 8);
    LODWORD(v50) = 64;
  }
  sub_C7D6A0((__int64)v49, 8 * v38, 8);
  v52 = ((((((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v50 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v50 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v50 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v50 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 16;
  v53 = (v52
       | (((((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v50 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v50 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v50 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v50 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(v7 + 24) = v53;
  v54 = (_QWORD *)sub_C7D670(8 * v53, 8);
  v55 = *(unsigned int *)(v7 + 24);
  *(_QWORD *)(v7 + 16) = 0;
  *(_QWORD *)(v7 + 8) = v54;
  for ( j = &v54[v55]; j != v54; ++v54 )
  {
    if ( v54 )
      *v54 = -4096;
  }
LABEL_27:
  v41 = (unsigned int)v60;
  v42 = v59;
  *(_DWORD *)(v7 + 40) = 0;
  v43 = &v42[3 * v41];
  if ( v42 != v43 )
  {
    do
    {
      v44 = *(v43 - 1);
      v43 -= 3;
      if ( v44 != 0 && v44 != -4096 && v44 != -8192 )
        sub_BD60C0(v43);
    }
    while ( v42 != v43 );
    v43 = v59;
  }
  if ( v43 != (_QWORD *)v61 )
    _libc_free((unsigned __int64)v43);
  return v36;
}
