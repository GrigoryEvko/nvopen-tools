// Function: sub_27B5650
// Address: 0x27b5650
//
__int64 __fastcall sub_27B5650(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *j; // rdx
  __int64 *v6; // r12
  __int64 result; // rax
  __int64 *v8; // r14
  _QWORD *v9; // rdi
  __int64 *v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  int v14; // eax
  unsigned int v15; // esi
  __int64 v16; // r9
  __int64 v17; // r8
  int v18; // r11d
  __int64 *v19; // r10
  unsigned int v20; // edx
  __int64 *v21; // rdi
  __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 *v26; // r14
  __int64 *v27; // r12
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // r15
  __int64 v35; // rax
  __int64 *v36; // r13
  __int64 *v37; // r15
  __int64 v38; // r8
  unsigned int v39; // eax
  __int64 *v40; // rdi
  __int64 v41; // rcx
  unsigned int v42; // esi
  __int64 *v43; // r10
  int v44; // edx
  unsigned int v45; // ecx
  unsigned int v46; // eax
  _QWORD *v47; // rdi
  __int64 v48; // r12
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // rax
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _QWORD *i; // rdx
  int v54; // r11d
  int v55; // eax
  _QWORD *v56; // rax
  const void *v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+20h] [rbp-40h] BYREF
  __int64 v59[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  *(_BYTE *)(a1 + 144) = 0;
  if ( v2 )
  {
    v45 = 4 * v2;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v2) < 0x40 )
      v45 = 64;
    if ( v45 >= (unsigned int)v3 )
      goto LABEL_4;
    v46 = v2 - 1;
    if ( v46 )
    {
      _BitScanReverse(&v46, v46);
      v47 = *(_QWORD **)(a1 + 8);
      v48 = (unsigned int)(1 << (33 - (v46 ^ 0x1F)));
      if ( (int)v48 < 64 )
        v48 = 64;
      if ( (_DWORD)v48 == (_DWORD)v3 )
      {
        *(_QWORD *)(a1 + 16) = 0;
        v56 = &v47[v48];
        do
        {
          if ( v47 )
            *v47 = -4096;
          ++v47;
        }
        while ( v56 != v47 );
        goto LABEL_7;
      }
    }
    else
    {
      v47 = *(_QWORD **)(a1 + 8);
      LODWORD(v48) = 64;
    }
    sub_C7D6A0((__int64)v47, 8 * v3, 8);
    v49 = ((((((((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v48 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v48 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v48 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v48 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 16;
    v50 = (v49
         | (((((((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v48 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v48 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v48 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v48 / 3u + 1) | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v48 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v48 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v50;
    v51 = (_QWORD *)sub_C7D670(8 * v50, 8);
    v52 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v51;
    for ( i = &v51[v52]; i != v51; ++v51 )
    {
      if ( v51 )
        *v51 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 20) )
  {
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 <= 0x40 )
    {
LABEL_4:
      v4 = *(_QWORD **)(a1 + 8);
      for ( j = &v4[v3]; j != v4; ++v4 )
        *v4 = -4096;
      *(_QWORD *)(a1 + 16) = 0;
      goto LABEL_7;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 8 * v3, 8);
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
  }
LABEL_7:
  v6 = *(__int64 **)(a1 + 80);
  result = *(_QWORD *)(a1 + 88);
  *(_DWORD *)(a1 + 40) = 0;
  v8 = &v6[result];
  if ( v6 == v8 )
  {
    *(_DWORD *)(a1 + 104) = 0;
LABEL_49:
    *(_BYTE *)(a1 + 144) = 1;
    return result;
  }
  v57 = (const void *)(a1 + 48);
  do
  {
    while ( 1 )
    {
      v13 = *v6;
      v14 = *(_DWORD *)(a1 + 16);
      v58 = *v6;
      if ( !v14 )
      {
        v9 = *(_QWORD **)(a1 + 32);
        v10 = &v9[*(unsigned int *)(a1 + 40)];
        if ( v10 != sub_27ABED0(v9, (__int64)v10, &v58) )
          goto LABEL_10;
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(a1 + 32, v57, v11 + 1, 8u, v11, v12);
          v10 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
        }
        *v10 = v13;
        v35 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
        *(_DWORD *)(a1 + 40) = v35;
        if ( (unsigned int)v35 <= 4 )
          goto LABEL_10;
        v36 = *(__int64 **)(a1 + 32);
        v37 = &v36[v35];
        while ( 1 )
        {
          v42 = *(_DWORD *)(a1 + 24);
          if ( !v42 )
            break;
          v38 = *(_QWORD *)(a1 + 8);
          v39 = (v42 - 1) & (((unsigned int)*v36 >> 9) ^ ((unsigned int)*v36 >> 4));
          v40 = (__int64 *)(v38 + 8LL * v39);
          v41 = *v40;
          if ( *v36 != *v40 )
          {
            v54 = 1;
            v43 = 0;
            while ( v41 != -4096 )
            {
              if ( v41 != -8192 || v43 )
                v40 = v43;
              v39 = (v42 - 1) & (v54 + v39);
              v41 = *(_QWORD *)(v38 + 8LL * v39);
              if ( *v36 == v41 )
                goto LABEL_42;
              ++v54;
              v43 = v40;
              v40 = (__int64 *)(v38 + 8LL * v39);
            }
            v55 = *(_DWORD *)(a1 + 16);
            if ( !v43 )
              v43 = v40;
            ++*(_QWORD *)a1;
            v44 = v55 + 1;
            v59[0] = (__int64)v43;
            if ( 4 * (v55 + 1) < 3 * v42 )
            {
              if ( v42 - *(_DWORD *)(a1 + 20) - v44 > v42 >> 3 )
                goto LABEL_68;
              goto LABEL_46;
            }
LABEL_45:
            v42 *= 2;
LABEL_46:
            sub_CF28B0(a1, v42);
            sub_D6B660(a1, v36, v59);
            v43 = (__int64 *)v59[0];
            v44 = *(_DWORD *)(a1 + 16) + 1;
LABEL_68:
            *(_DWORD *)(a1 + 16) = v44;
            if ( *v43 != -4096 )
              --*(_DWORD *)(a1 + 20);
            *v43 = *v36;
          }
LABEL_42:
          if ( v37 == ++v36 )
            goto LABEL_10;
        }
        ++*(_QWORD *)a1;
        v59[0] = 0;
        goto LABEL_45;
      }
      v15 = *(_DWORD *)(a1 + 24);
      if ( !v15 )
      {
        ++*(_QWORD *)a1;
        v59[0] = 0;
        goto LABEL_73;
      }
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = 1;
      v19 = 0;
      v20 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v21 = (__int64 *)(v17 + 8LL * v20);
      v22 = *v21;
      if ( v13 != *v21 )
        break;
LABEL_10:
      if ( v8 == ++v6 )
        goto LABEL_24;
    }
    while ( v22 != -4096 )
    {
      if ( v22 != -8192 || v19 )
        v21 = v19;
      v20 = v16 & (v18 + v20);
      v22 = *(_QWORD *)(v17 + 8LL * v20);
      if ( v13 == v22 )
        goto LABEL_10;
      ++v18;
      v19 = v21;
      v21 = (__int64 *)(v17 + 8LL * v20);
    }
    if ( !v19 )
      v19 = v21;
    v23 = v14 + 1;
    ++*(_QWORD *)a1;
    v59[0] = (__int64)v19;
    if ( 4 * v23 < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a1 + 20) - v23 > v15 >> 3 )
        goto LABEL_19;
      goto LABEL_74;
    }
LABEL_73:
    v15 *= 2;
LABEL_74:
    sub_CF28B0(a1, v15);
    sub_D6B660(a1, &v58, v59);
    v13 = v58;
    v19 = (__int64 *)v59[0];
    v23 = *(_DWORD *)(a1 + 16) + 1;
LABEL_19:
    *(_DWORD *)(a1 + 16) = v23;
    if ( *v19 != -4096 )
      --*(_DWORD *)(a1 + 20);
    *v19 = v13;
    v24 = *(unsigned int *)(a1 + 40);
    v25 = v58;
    if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, v57, v24 + 1, 8u, v17, v16);
      v24 = *(unsigned int *)(a1 + 40);
    }
    ++v6;
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v24) = v25;
    ++*(_DWORD *)(a1 + 40);
  }
  while ( v8 != v6 );
LABEL_24:
  v26 = *(__int64 **)(a1 + 80);
  result = *(_QWORD *)(a1 + 88);
  *(_DWORD *)(a1 + 104) = 0;
  v27 = &v26[result];
  if ( v27 == v26 )
    goto LABEL_49;
  do
  {
    while ( 1 )
    {
      v59[0] = *v26;
      v29 = *(_QWORD *)(v59[0] + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v29 == v59[0] + 48 )
      {
        v31 = 0;
      }
      else
      {
        if ( !v29 )
          BUG();
        v30 = *(unsigned __int8 *)(v29 - 24);
        v31 = v29 - 24;
        if ( (unsigned int)(v30 - 30) >= 0xB )
          v31 = 0;
      }
      v34 = sub_B46BC0(v31, 0);
      if ( v34 )
        break;
      ++v26;
      sub_27AC510(a1, v59);
      if ( v27 == v26 )
        goto LABEL_35;
    }
    v28 = *(unsigned int *)(a1 + 104);
    if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 108) )
    {
      sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), v28 + 1, 8u, v32, v33);
      v28 = *(unsigned int *)(a1 + 104);
    }
    ++v26;
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v28) = v34;
    ++*(_DWORD *)(a1 + 104);
  }
  while ( v27 != v26 );
LABEL_35:
  result = *(unsigned int *)(a1 + 104);
  if ( !(_DWORD)result )
    goto LABEL_49;
  return result;
}
