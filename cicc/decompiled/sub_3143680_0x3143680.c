// Function: sub_3143680
// Address: 0x3143680
//
__int64 __fastcall sub_3143680(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v3; // r14
  unsigned __int8 *v5; // r12
  unsigned int v7; // r10d
  unsigned int v8; // r15d
  unsigned int v9; // ecx
  unsigned __int8 **v10; // rax
  unsigned __int8 *v11; // rdx
  unsigned __int8 **v12; // r11
  unsigned __int8 **v13; // r9
  unsigned int v14; // esi
  unsigned __int8 **v15; // rdi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // eax
  unsigned __int8 **v20; // r8
  unsigned __int8 *v21; // rsi
  int v22; // edx
  int v23; // r10d
  unsigned __int8 **v24; // r9
  unsigned __int8 *v25; // r8
  unsigned int v26; // r9d
  int v27; // r11d
  int v28; // eax
  unsigned int v29; // edx
  int v31; // r9d
  int v32; // eax
  int v33; // eax
  unsigned int v34; // ecx
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *k; // rdx
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  int v41; // r9d
  unsigned __int8 **v42; // rdi
  unsigned int v43; // r15d
  unsigned __int8 *v44; // rcx
  unsigned int v45; // eax
  int v46; // eax
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rax
  int v49; // r13d
  __int64 v50; // r12
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _QWORD *i; // rdx
  unsigned int v54; // eax
  _QWORD *v55; // rdi
  __int64 v56; // r12
  _QWORD *v57; // rax
  unsigned __int64 v58; // rdx
  unsigned __int64 v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // rdx
  _QWORD *j; // rdx
  int v63; // [rsp-3Ch] [rbp-3Ch]

  if ( !a2 )
    return 0;
  v3 = a1 + 64;
  v5 = a2;
  while ( 1 )
  {
    v14 = *(_DWORD *)(a1 + 88);
    v15 = *(unsigned __int8 ***)(a1 + 72);
    if ( !v14 )
    {
      ++*(_QWORD *)(a1 + 64);
LABEL_10:
      sub_31434B0(v3, 2 * v14);
      v16 = *(_DWORD *)(a1 + 88);
      if ( !v16 )
        goto LABEL_106;
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 72);
      v19 = (v16 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v20 = (unsigned __int8 **)(v18 + 8LL * v19);
      v21 = *v20;
      v22 = *(_DWORD *)(a1 + 80) + 1;
      if ( v5 != *v20 )
      {
        v23 = 1;
        v24 = 0;
        while ( v21 != (unsigned __int8 *)-4096LL )
        {
          if ( v21 == (unsigned __int8 *)-8192LL && !v24 )
            v24 = v20;
          v19 = v17 & (v23 + v19);
          v20 = (unsigned __int8 **)(v18 + 8LL * v19);
          v21 = *v20;
          if ( v5 == *v20 )
            goto LABEL_34;
          ++v23;
        }
        if ( v24 )
          v20 = v24;
      }
      goto LABEL_34;
    }
    v7 = v14 - 1;
    v8 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
    v9 = (v14 - 1) & v8;
    v10 = &v15[v9];
    v11 = *v10;
    v12 = v10;
    if ( v5 == *v10 )
    {
LABEL_4:
      v13 = &v15[v14];
      if ( v13 == v12 )
        goto LABEL_5;
      v28 = *(_DWORD *)(a1 + 80);
      ++*(_QWORD *)(a1 + 64);
      if ( v28 )
      {
        v29 = 4 * v28;
        if ( (unsigned int)(4 * v28) < 0x40 )
          v29 = 64;
        if ( v14 <= v29 )
        {
          do
LABEL_24:
            *v15++ = (unsigned __int8 *)-4096LL;
          while ( v13 != v15 );
          *(_QWORD *)(a1 + 80) = 0;
          return 0;
        }
        v45 = v28 - 1;
        if ( v45 )
        {
          _BitScanReverse(&v45, v45);
          v46 = 1 << (33 - (v45 ^ 0x1F));
          if ( v46 < 64 )
            v46 = 64;
          if ( v14 == v46 )
          {
            *(_QWORD *)(a1 + 80) = 0;
            do
            {
              if ( v15 )
                *v15 = (unsigned __int8 *)-4096LL;
              ++v15;
            }
            while ( v13 != v15 );
            return 0;
          }
          v47 = (4 * v46 / 3u + 1) | ((unsigned __int64)(4 * v46 / 3u + 1) >> 1);
          v48 = ((v47 | (v47 >> 2)) >> 4) | v47 | (v47 >> 2) | ((((v47 | (v47 >> 2)) >> 4) | v47 | (v47 >> 2)) >> 8);
          v49 = (v48 | (v48 >> 16)) + 1;
          v50 = 8 * ((v48 | (v48 >> 16)) + 1);
        }
        else
        {
          v50 = 1024;
          v49 = 128;
        }
        sub_C7D6A0((__int64)v15, 8LL * v14, 8);
        *(_DWORD *)(a1 + 88) = v49;
        v51 = (_QWORD *)sub_C7D670(v50, 8);
        v52 = *(unsigned int *)(a1 + 88);
        *(_QWORD *)(a1 + 80) = 0;
        *(_QWORD *)(a1 + 72) = v51;
        for ( i = &v51[v52]; i != v51; ++v51 )
        {
          if ( v51 )
            *v51 = -4096;
        }
      }
      else if ( *(_DWORD *)(a1 + 84) )
      {
        if ( v14 > 0x40 )
        {
          sub_C7D6A0((__int64)v15, 8LL * v14, 8);
          *(_QWORD *)(a1 + 72) = 0;
          *(_QWORD *)(a1 + 80) = 0;
          *(_DWORD *)(a1 + 88) = 0;
          return 0;
        }
        goto LABEL_24;
      }
      return 0;
    }
    v25 = *v10;
    v26 = (v14 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v27 = 1;
    while ( v25 != (unsigned __int8 *)-4096LL )
    {
      v26 = v7 & (v27 + v26);
      v63 = v27 + 1;
      v12 = &v15[v26];
      v25 = *v12;
      if ( *v12 == v5 )
        goto LABEL_4;
      v27 = v63;
    }
    v8 = ((unsigned int)v5 >> 4) ^ ((unsigned int)v5 >> 9);
    v9 = v7 & v8;
    v10 = &v15[v7 & v8];
    v11 = *v10;
LABEL_5:
    if ( v5 == v11 )
    {
LABEL_6:
      if ( a3 == v5 )
        break;
      goto LABEL_7;
    }
    v31 = 1;
    v20 = 0;
    while ( v11 != (unsigned __int8 *)-4096LL )
    {
      if ( v20 || v11 != (unsigned __int8 *)-8192LL )
        v10 = v20;
      v9 = v7 & (v31 + v9);
      v11 = v15[v9];
      if ( v11 == v5 )
        goto LABEL_6;
      ++v31;
      v20 = v10;
      v10 = &v15[v9];
    }
    if ( !v20 )
      v20 = v10;
    v32 = *(_DWORD *)(a1 + 80);
    ++*(_QWORD *)(a1 + 64);
    v22 = v32 + 1;
    if ( 4 * (v32 + 1) >= 3 * v14 )
      goto LABEL_10;
    if ( v14 - *(_DWORD *)(a1 + 84) - v22 <= v14 >> 3 )
    {
      sub_31434B0(v3, v14);
      v38 = *(_DWORD *)(a1 + 88);
      if ( !v38 )
      {
LABEL_106:
        ++*(_DWORD *)(a1 + 80);
        BUG();
      }
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 72);
      v41 = 1;
      v42 = 0;
      v43 = v39 & v8;
      v20 = (unsigned __int8 **)(v40 + 8LL * v43);
      v44 = *v20;
      v22 = *(_DWORD *)(a1 + 80) + 1;
      if ( v5 != *v20 )
      {
        while ( v44 != (unsigned __int8 *)-4096LL )
        {
          if ( v44 == (unsigned __int8 *)-8192LL && !v42 )
            v42 = v20;
          v43 = v39 & (v41 + v43);
          v20 = (unsigned __int8 **)(v40 + 8LL * v43);
          v44 = *v20;
          if ( v5 == *v20 )
            goto LABEL_34;
          ++v41;
        }
        if ( v42 )
          v20 = v42;
      }
    }
LABEL_34:
    *(_DWORD *)(a1 + 80) = v22;
    if ( *v20 != (unsigned __int8 *)-4096LL )
      --*(_DWORD *)(a1 + 84);
    *v20 = v5;
    if ( a3 == v5 )
      break;
LABEL_7:
    v5 = (unsigned __int8 *)sub_AF2660(v5);
    if ( !v5 )
      return 0;
  }
  v33 = *(_DWORD *)(a1 + 80);
  ++*(_QWORD *)(a1 + 64);
  if ( !v33 )
  {
    if ( *(_DWORD *)(a1 + 84) )
    {
      v35 = *(unsigned int *)(a1 + 88);
      if ( (unsigned int)v35 <= 0x40 )
        goto LABEL_41;
      sub_C7D6A0(*(_QWORD *)(a1 + 72), 8 * v35, 8);
      *(_QWORD *)(a1 + 72) = 0;
      *(_QWORD *)(a1 + 80) = 0;
      *(_DWORD *)(a1 + 88) = 0;
    }
    return 1;
  }
  v34 = 4 * v33;
  v35 = *(unsigned int *)(a1 + 88);
  if ( (unsigned int)(4 * v33) < 0x40 )
    v34 = 64;
  if ( v34 < (unsigned int)v35 )
  {
    v54 = v33 - 1;
    if ( v54 )
    {
      _BitScanReverse(&v54, v54);
      v55 = *(_QWORD **)(a1 + 72);
      v56 = (unsigned int)(1 << (33 - (v54 ^ 0x1F)));
      if ( (int)v56 < 64 )
        v56 = 64;
      if ( (_DWORD)v56 == (_DWORD)v35 )
      {
        *(_QWORD *)(a1 + 80) = 0;
        v57 = &v55[v56];
        do
        {
          if ( v55 )
            *v55 = -4096;
          ++v55;
        }
        while ( v57 != v55 );
        return 1;
      }
    }
    else
    {
      v55 = *(_QWORD **)(a1 + 72);
      LODWORD(v56) = 64;
    }
    sub_C7D6A0((__int64)v55, 8 * v35, 8);
    v58 = ((((((((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v56 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v56 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v56 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v56 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 16;
    v59 = (v58
         | (((((((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v56 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v56 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v56 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v56 / 3u + 1) | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v56 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v56 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 88) = v59;
    v60 = (_QWORD *)sub_C7D670(8 * v59, 8);
    v61 = *(unsigned int *)(a1 + 88);
    *(_QWORD *)(a1 + 80) = 0;
    *(_QWORD *)(a1 + 72) = v60;
    for ( j = &v60[v61]; j != v60; ++v60 )
    {
      if ( v60 )
        *v60 = -4096;
    }
    return 1;
  }
LABEL_41:
  v36 = *(_QWORD **)(a1 + 72);
  for ( k = &v36[v35]; k != v36; ++v36 )
    *v36 = -4096;
  *(_QWORD *)(a1 + 80) = 0;
  return 1;
}
