// Function: sub_376D860
// Address: 0x376d860
//
unsigned __int64 __fastcall sub_376D860(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4)
{
  int v6; // r12d
  int v7; // r15d
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // rcx
  int v11; // r14d
  char v12; // dl
  __int64 v13; // r9
  int v14; // esi
  int v15; // r11d
  __int64 v16; // rdi
  unsigned int i; // r10d
  __int64 v18; // rax
  unsigned int v19; // r10d
  unsigned int v20; // esi
  char v21; // dl
  __int64 v22; // r9
  int v23; // esi
  int v24; // r11d
  __int64 v25; // rdi
  unsigned int v26; // r10d
  __int64 *v27; // rax
  __int64 v28; // r8
  unsigned int v29; // r10d
  unsigned int v30; // eax
  int v31; // r8d
  unsigned int v32; // r9d
  unsigned int v34; // esi
  unsigned int v35; // eax
  int v36; // r8d
  unsigned int v37; // r9d
  int v38; // r8d
  int v39; // eax
  __int64 v40; // rsi
  int v41; // eax
  int v42; // r10d
  __int64 v43; // r9
  unsigned int v44; // r8d
  unsigned int v45; // r8d
  int v46; // eax
  __int64 v47; // rsi
  int v48; // eax
  int v49; // r10d
  __int64 v50; // r9
  unsigned int k; // r8d
  unsigned int v52; // r8d
  int v53; // eax
  __int64 v54; // rsi
  int v55; // eax
  int v56; // r10d
  unsigned int j; // r8d
  unsigned int v58; // r8d
  int v59; // eax
  __int64 v60; // rsi
  int v61; // eax
  int v62; // r10d
  unsigned int m; // r8d
  unsigned int v64; // r8d
  int v65; // r8d
  int v66; // edx
  int v67; // edx
  int v68; // edx
  int v69; // edx
  unsigned __int64 v70; // [rsp+8h] [rbp-48h]
  unsigned __int64 v71; // [rsp+8h] [rbp-48h]
  unsigned __int64 v72; // [rsp+8h] [rbp-48h]
  unsigned __int64 v73; // [rsp+8h] [rbp-48h]
  const __m128i *v74; // [rsp+10h] [rbp-40h]
  int v75; // [rsp+1Ch] [rbp-34h]

  v75 = *(_DWORD *)(a2 + 68);
  if ( !v75 )
    return a4;
  v6 = 0;
  v7 = (a2 >> 9) ^ (a2 >> 4);
  v9 = a4 >> 4;
  v10 = a2;
  v74 = (const __m128i *)(a1 + 24);
  v11 = (a4 >> 9) ^ v9;
  do
  {
    v12 = *(_BYTE *)(a1 + 32) & 1;
    if ( v12 )
    {
      v13 = a1 + 40;
      v14 = 63;
    }
    else
    {
      v20 = *(_DWORD *)(a1 + 48);
      v13 = *(_QWORD *)(a1 + 40);
      if ( !v20 )
      {
        v35 = *(_DWORD *)(a1 + 32);
        ++*(_QWORD *)(a1 + 24);
        v16 = 0;
        v36 = (v35 >> 1) + 1;
LABEL_31:
        v37 = 3 * v20;
        goto LABEL_32;
      }
      v14 = v20 - 1;
    }
    v15 = 1;
    v16 = 0;
    for ( i = v14 & v7; ; i = v14 & v19 )
    {
      v18 = v13 + 32LL * i;
      if ( v10 != *(_QWORD *)v18 )
        break;
      if ( *(_DWORD *)(v18 + 8) == v6 )
        goto LABEL_12;
LABEL_8:
      v19 = v15 + i;
      ++v15;
    }
    if ( *(_QWORD *)v18 )
      goto LABEL_8;
    v65 = *(_DWORD *)(v18 + 8);
    if ( v65 != -1 )
    {
      if ( v65 == -2 && !v16 )
        v16 = v13 + 32LL * i;
      goto LABEL_8;
    }
    if ( !v16 )
      v16 = v13 + 32LL * i;
    v35 = *(_DWORD *)(a1 + 32);
    ++*(_QWORD *)(a1 + 24);
    v36 = (v35 >> 1) + 1;
    if ( !v12 )
    {
      v20 = *(_DWORD *)(a1 + 48);
      goto LABEL_31;
    }
    v37 = 192;
    v20 = 64;
LABEL_32:
    if ( 4 * v36 < v37 )
    {
      if ( v20 - *(_DWORD *)(a1 + 36) - v36 > v20 >> 3 )
        goto LABEL_34;
      v72 = v10;
      sub_376D3B0(v74, v20);
      v10 = v72;
      if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
      {
        v54 = a1 + 40;
        v55 = 63;
        goto LABEL_65;
      }
      v53 = *(_DWORD *)(a1 + 48);
      v54 = *(_QWORD *)(a1 + 40);
      if ( v53 )
      {
        v55 = v53 - 1;
LABEL_65:
        v56 = 1;
        v43 = 0;
        for ( j = v7 & v55; ; j = v55 & v58 )
        {
          v16 = v54 + 32LL * j;
          if ( v72 == *(_QWORD *)v16 )
          {
            if ( v6 == *(_DWORD *)(v16 + 8) )
              goto LABEL_90;
          }
          else if ( !*(_QWORD *)v16 )
          {
            v67 = *(_DWORD *)(v16 + 8);
            if ( v67 == -1 )
              goto LABEL_121;
            if ( !v43 && v67 == -2 )
              v43 = v54 + 32LL * j;
          }
          v58 = v56 + j;
          ++v56;
        }
      }
LABEL_128:
      *(_DWORD *)(a1 + 32) = (2 * (*(_DWORD *)(a1 + 32) >> 1) + 2) | *(_DWORD *)(a1 + 32) & 1;
      BUG();
    }
    v70 = v10;
    sub_376D3B0(v74, 2 * v20);
    v10 = v70;
    if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
    {
      v40 = a1 + 40;
      v41 = 63;
    }
    else
    {
      v39 = *(_DWORD *)(a1 + 48);
      v40 = *(_QWORD *)(a1 + 40);
      if ( !v39 )
        goto LABEL_128;
      v41 = v39 - 1;
    }
    v42 = 1;
    v43 = 0;
    v44 = v7 & v41;
    while ( 2 )
    {
      v16 = v40 + 32LL * v44;
      if ( v70 == *(_QWORD *)v16 )
      {
        if ( v6 == *(_DWORD *)(v16 + 8) )
          goto LABEL_90;
        goto LABEL_54;
      }
      if ( *(_QWORD *)v16 )
      {
LABEL_54:
        v45 = v42 + v44;
        ++v42;
        v44 = v41 & v45;
        continue;
      }
      break;
    }
    v66 = *(_DWORD *)(v16 + 8);
    if ( v66 != -1 )
    {
      if ( v66 == -2 && !v43 )
        v43 = v40 + 32LL * v44;
      goto LABEL_54;
    }
LABEL_121:
    if ( v43 )
      v16 = v43;
LABEL_90:
    v35 = *(_DWORD *)(a1 + 32);
LABEL_34:
    *(_DWORD *)(a1 + 32) = (2 * (v35 >> 1) + 2) | v35 & 1;
    if ( *(_QWORD *)v16 || *(_DWORD *)(v16 + 8) != -1 )
      --*(_DWORD *)(a1 + 36);
    *(_QWORD *)v16 = v10;
    *(_DWORD *)(v16 + 8) = v6;
    *(_QWORD *)(v16 + 16) = a4;
    *(_DWORD *)(v16 + 24) = v6;
LABEL_12:
    if ( a4 == v10 )
      goto LABEL_26;
    v21 = *(_BYTE *)(a1 + 32) & 1;
    if ( v21 )
    {
      v22 = a1 + 40;
      v23 = 63;
      goto LABEL_15;
    }
    v34 = *(_DWORD *)(a1 + 48);
    v22 = *(_QWORD *)(a1 + 40);
    if ( !v34 )
    {
      v30 = *(_DWORD *)(a1 + 32);
      ++*(_QWORD *)(a1 + 24);
      v25 = 0;
      v31 = (v30 >> 1) + 1;
LABEL_20:
      v32 = 3 * v34;
      goto LABEL_21;
    }
    v23 = v34 - 1;
LABEL_15:
    v24 = 1;
    v25 = 0;
    v26 = v11 & v23;
    while ( 2 )
    {
      v27 = (__int64 *)(v22 + 32LL * v26);
      v28 = *v27;
      if ( a4 == *v27 )
      {
        if ( v6 == *((_DWORD *)v27 + 2) )
          goto LABEL_26;
        if ( v28 )
          goto LABEL_18;
      }
      else if ( v28 )
      {
LABEL_18:
        v29 = v24 + v26;
        ++v24;
        v26 = v23 & v29;
        continue;
      }
      break;
    }
    v38 = *((_DWORD *)v27 + 2);
    if ( v38 != -1 )
    {
      if ( !v25 && v38 == -2 )
        v25 = v22 + 32LL * v26;
      goto LABEL_18;
    }
    if ( !v25 )
      v25 = v22 + 32LL * v26;
    v30 = *(_DWORD *)(a1 + 32);
    ++*(_QWORD *)(a1 + 24);
    v31 = (v30 >> 1) + 1;
    if ( !v21 )
    {
      v34 = *(_DWORD *)(a1 + 48);
      goto LABEL_20;
    }
    v32 = 192;
    v34 = 64;
LABEL_21:
    if ( v32 <= 4 * v31 )
    {
      v71 = v10;
      sub_376D3B0(v74, 2 * v34);
      v10 = v71;
      if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
      {
        v47 = a1 + 40;
        v48 = 63;
        goto LABEL_58;
      }
      v46 = *(_DWORD *)(a1 + 48);
      v47 = *(_QWORD *)(a1 + 40);
      if ( v46 )
      {
        v48 = v46 - 1;
LABEL_58:
        v49 = 1;
        v50 = 0;
        for ( k = v11 & v48; ; k = v48 & v52 )
        {
          v25 = v47 + 32LL * k;
          if ( a4 == *(_QWORD *)v25 && v6 == *(_DWORD *)(v25 + 8) )
            break;
          if ( !*(_QWORD *)v25 )
          {
            v69 = *(_DWORD *)(v25 + 8);
            if ( v69 == -1 )
            {
LABEL_123:
              if ( v50 )
                v25 = v50;
              break;
            }
            if ( !v50 && v69 == -2 )
              v50 = v47 + 32LL * k;
          }
          v52 = v49 + k;
          ++v49;
        }
LABEL_98:
        v30 = *(_DWORD *)(a1 + 32);
        goto LABEL_23;
      }
LABEL_127:
      *(_DWORD *)(a1 + 32) = (2 * (*(_DWORD *)(a1 + 32) >> 1) + 2) | *(_DWORD *)(a1 + 32) & 1;
      BUG();
    }
    if ( v34 - *(_DWORD *)(a1 + 36) - v31 <= v34 >> 3 )
    {
      v73 = v10;
      sub_376D3B0(v74, v34);
      v10 = v73;
      if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
      {
        v60 = a1 + 40;
        v61 = 63;
      }
      else
      {
        v59 = *(_DWORD *)(a1 + 48);
        v60 = *(_QWORD *)(a1 + 40);
        if ( !v59 )
          goto LABEL_127;
        v61 = v59 - 1;
      }
      v62 = 1;
      v50 = 0;
      for ( m = v11 & v61; ; m = v61 & v64 )
      {
        v25 = v60 + 32LL * m;
        if ( a4 == *(_QWORD *)v25 && v6 == *(_DWORD *)(v25 + 8) )
          break;
        if ( !*(_QWORD *)v25 )
        {
          v68 = *(_DWORD *)(v25 + 8);
          if ( v68 == -1 )
            goto LABEL_123;
          if ( v68 == -2 && !v50 )
            v50 = v60 + 32LL * m;
        }
        v64 = v62 + m;
        ++v62;
      }
      goto LABEL_98;
    }
LABEL_23:
    *(_DWORD *)(a1 + 32) = (2 * (v30 >> 1) + 2) | v30 & 1;
    if ( *(_QWORD *)v25 || *(_DWORD *)(v25 + 8) != -1 )
      --*(_DWORD *)(a1 + 36);
    *(_QWORD *)v25 = a4;
    *(_DWORD *)(v25 + 8) = v6;
    *(_QWORD *)(v25 + 16) = a4;
    *(_DWORD *)(v25 + 24) = v6;
LABEL_26:
    ++v6;
    ++v7;
    ++v11;
  }
  while ( v75 != v6 );
  return a4;
}
