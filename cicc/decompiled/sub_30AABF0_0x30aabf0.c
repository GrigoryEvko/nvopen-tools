// Function: sub_30AABF0
// Address: 0x30aabf0
//
__int64 __fastcall sub_30AABF0(unsigned int **a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int *v3; // rdx
  __int64 v4; // r8
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r15
  unsigned int *v12; // rax
  unsigned __int64 v13; // rcx
  _QWORD *v14; // rbx
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rsi
  _BYTE *v21; // rax
  _BYTE *v22; // r14
  __int64 v23; // r11
  __int64 v24; // r8
  unsigned int v25; // esi
  __int64 v26; // r9
  __int64 v27; // r10
  unsigned int v28; // ecx
  __int64 *v29; // rdi
  unsigned int i; // edx
  __int64 *v31; // rax
  __int64 v32; // rcx
  unsigned int v33; // edx
  int v34; // edi
  int v35; // edx
  __int64 v36; // rax
  __int64 *v37; // rax
  int v38; // esi
  int v39; // esi
  __int64 v40; // rdi
  unsigned int j; // edx
  __int64 *v42; // rcx
  __int64 v43; // r10
  int v44; // edx
  int v45; // edx
  __int64 v46; // rcx
  int v47; // edi
  int v48; // esi
  unsigned int v49; // edx
  unsigned int v50; // edx
  __int64 v51; // [rsp-50h] [rbp-50h]
  unsigned __int64 v52; // [rsp-48h] [rbp-48h]
  __int64 v53; // [rsp-48h] [rbp-48h]
  __int64 v54; // [rsp-48h] [rbp-48h]
  int v55; // [rsp-40h] [rbp-40h]
  __int64 v56; // [rsp-40h] [rbp-40h]
  __int64 v57; // [rsp-40h] [rbp-40h]
  __int64 v58; // [rsp-40h] [rbp-40h]

  result = *(_QWORD *)(a2 + 184);
  v3 = *a1;
  if ( !result )
    return result;
  v4 = a2 + 176;
  v6 = a2 + 176;
  v7 = *v3;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(result + 16);
      v9 = *(_QWORD *)(result + 24);
      if ( *(_DWORD *)(result + 32) >= v7 )
        break;
      result = *(_QWORD *)(result + 24);
      if ( !v9 )
        goto LABEL_6;
    }
    v6 = result;
    result = *(_QWORD *)(result + 16);
  }
  while ( v8 );
LABEL_6:
  if ( v4 == v6 )
    return result;
  if ( v7 < *(_DWORD *)(v6 + 32) )
    return result;
  v10 = *(_QWORD *)(v6 + 64);
  v11 = v6 + 48;
  if ( v10 == v6 + 48 )
    return result;
  do
  {
    while ( 1 )
    {
      v12 = a1[1];
      v13 = *(_QWORD *)(v10 + 32);
      v14 = (_QWORD *)*((_QWORD *)v12 + 14);
      v15 = v12 + 26;
      if ( v14 )
      {
        v16 = v12 + 26;
        do
        {
          while ( 1 )
          {
            v17 = v14[2];
            v18 = v14[3];
            if ( v13 <= v14[4] )
              break;
            v14 = (_QWORD *)v14[3];
            if ( !v18 )
              goto LABEL_14;
          }
          v16 = v14;
          v14 = (_QWORD *)v14[2];
        }
        while ( v17 );
LABEL_14:
        if ( v15 != v16 && v13 >= v16[4] )
        {
          v19 = v16[7];
          v20 = v16[6];
          if ( v19 )
          {
            v21 = sub_BA8CB0((__int64)a1[2], v20, v19);
            v22 = v21;
            if ( v21 )
            {
              if ( (unsigned __int8)sub_B2D610((__int64)v21, 3) )
                break;
            }
          }
        }
      }
LABEL_19:
      result = sub_220EF30(v10);
      v10 = result;
      if ( v11 == result )
        return result;
    }
    v23 = (__int64)a1[3];
    v24 = (__int64)a1[4];
    v25 = *(_DWORD *)(v23 + 24);
    if ( !v25 )
    {
      ++*(_QWORD *)v23;
      goto LABEL_55;
    }
    v55 = 1;
    v26 = v25 - 1;
    v27 = *(_QWORD *)(v23 + 8);
    v28 = (unsigned int)v22 >> 9;
    v29 = 0;
    v52 = ((0xBF58476D1CE4E5B9LL
          * (v28 ^ ((unsigned int)v22 >> 4)
           | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))) >> 31)
        ^ (0xBF58476D1CE4E5B9LL
         * (v28 ^ ((unsigned int)v22 >> 4)
          | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32)));
    for ( i = v26
            & (((0xBF58476D1CE4E5B9LL
               * (v28 ^ ((unsigned int)v22 >> 4)
                | ((unsigned __int64)(((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)) << 32))) >> 31)
             ^ (484763065 * (v28 ^ ((unsigned int)v22 >> 4)))); ; i = v26 & v33 )
    {
      v31 = (__int64 *)(v27 + 16LL * i);
      v32 = *v31;
      if ( v24 == *v31 && v22 == (_BYTE *)v31[1] )
        goto LABEL_19;
      if ( v32 == -4096 )
        break;
      if ( v32 == -8192 && v31[1] == -8192 && !v29 )
        v29 = (__int64 *)(v27 + 16LL * i);
LABEL_30:
      v33 = v55 + i;
      ++v55;
    }
    if ( v31[1] != -4096 )
      goto LABEL_30;
    if ( v29 )
      v31 = v29;
    v34 = *(_DWORD *)(v23 + 16);
    ++*(_QWORD *)v23;
    v35 = v34 + 1;
    if ( 4 * (v34 + 1) < 3 * v25 )
    {
      if ( v25 - *(_DWORD *)(v23 + 20) - v35 > v25 >> 3 )
        goto LABEL_37;
      v56 = v23;
      v51 = v24;
      sub_30AA930(v23, v25);
      v23 = v56;
      v38 = *(_DWORD *)(v56 + 24);
      if ( v38 )
      {
        v39 = v38 - 1;
        v31 = 0;
        v24 = v51;
        v26 = 1;
        for ( j = v39 & v52; ; j = v39 & v44 )
        {
          v40 = *(_QWORD *)(v56 + 8);
          v42 = (__int64 *)(v40 + 16LL * j);
          v43 = *v42;
          if ( v51 == *v42 && v22 == (_BYTE *)v42[1] )
          {
            v35 = *(_DWORD *)(v56 + 16) + 1;
            v31 = v42;
            goto LABEL_37;
          }
          if ( v43 == -4096 )
          {
            if ( v42[1] == -4096 )
            {
              if ( !v31 )
                v31 = (__int64 *)(v40 + 16LL * j);
              v35 = *(_DWORD *)(v56 + 16) + 1;
              goto LABEL_37;
            }
          }
          else if ( v43 == -8192 && v42[1] == -8192 && !v31 )
          {
            v31 = (__int64 *)(v40 + 16LL * j);
          }
          v44 = v26 + j;
          v26 = (unsigned int)(v26 + 1);
        }
      }
LABEL_76:
      ++*(_DWORD *)(v23 + 16);
      BUG();
    }
LABEL_55:
    v57 = v23;
    v53 = v24;
    sub_30AA930(v23, 2 * v25);
    v23 = v57;
    v45 = *(_DWORD *)(v57 + 24);
    if ( !v45 )
      goto LABEL_76;
    v24 = v53;
    v47 = v45 - 1;
    v48 = 1;
    v49 = (v45 - 1)
        & (((0xBF58476D1CE4E5B9LL
           * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)
            | ((unsigned __int64)(((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4)) << 32))) >> 31)
         ^ (484763065 * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4))));
    while ( 2 )
    {
      v46 = *(_QWORD *)(v57 + 8);
      v31 = (__int64 *)(v46 + 16LL * v49);
      v26 = *v31;
      if ( v53 == *v31 && v22 == (_BYTE *)v31[1] )
      {
        v35 = *(_DWORD *)(v57 + 16) + 1;
        goto LABEL_37;
      }
      if ( v26 != -4096 )
      {
        if ( v26 == -8192 && v31[1] == -8192 && !v14 )
          v14 = (_QWORD *)(v46 + 16LL * v49);
        goto LABEL_63;
      }
      if ( v31[1] != -4096 )
      {
LABEL_63:
        v50 = v48 + v49;
        ++v48;
        v49 = v47 & v50;
        continue;
      }
      break;
    }
    if ( v14 )
      v31 = v14;
    v35 = *(_DWORD *)(v57 + 16) + 1;
LABEL_37:
    *(_DWORD *)(v23 + 16) = v35;
    if ( *v31 != -4096 || v31[1] != -4096 )
      --*(_DWORD *)(v23 + 20);
    *v31 = v24;
    v31[1] = (__int64)v22;
    v36 = *(unsigned int *)(v23 + 40);
    if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 44) )
    {
      v54 = v24;
      v58 = v23;
      sub_C8D5F0(v23 + 32, (const void *)(v23 + 48), v36 + 1, 0x10u, v24, v26);
      v23 = v58;
      v24 = v54;
      v36 = *(unsigned int *)(v58 + 40);
    }
    v37 = (__int64 *)(*(_QWORD *)(v23 + 32) + 16 * v36);
    *v37 = v24;
    v37[1] = (__int64)v22;
    ++*(_DWORD *)(v23 + 40);
    result = sub_220EF30(v10);
    v10 = result;
  }
  while ( v11 != result );
  return result;
}
