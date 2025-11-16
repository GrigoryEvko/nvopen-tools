// Function: sub_FD0080
// Address: 0xfd0080
//
__int64 __fastcall sub_FD0080(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 result; // rax
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r13
  unsigned int v19; // esi
  __int64 v20; // rdx
  unsigned int v21; // r8d
  unsigned int v22; // r14d
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r11
  int v30; // r11d
  __int64 v31; // r11
  __int64 *v32; // r11
  int v33; // eax
  int v34; // eax
  int v35; // eax
  int v36; // eax
  int v37; // edi
  __int64 v38; // r8
  unsigned int v39; // r14d
  __int64 v40; // rsi
  int v41; // ecx
  __int64 *v42; // rdx
  int v43; // eax
  int v44; // edi
  __int64 v45; // r8
  unsigned int v46; // r14d
  int v47; // ecx
  __int64 v48; // rsi
  int v49; // r10d
  int v50; // [rsp+10h] [rbp-50h]
  unsigned int v51; // [rsp+18h] [rbp-48h]
  int v52; // [rsp+18h] [rbp-48h]
  __int64 v53; // [rsp+18h] [rbp-48h]
  __int64 v54; // [rsp+18h] [rbp-48h]
  __int64 v55; // [rsp+20h] [rbp-40h]
  __int64 v56; // [rsp+28h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 136);
  v7 = *(_QWORD *)(a1 + 120);
  if ( v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_3;
    v35 = 1;
    while ( v10 != -4096 )
    {
      v49 = v35 + 1;
      v8 = (v6 - 1) & (v35 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v35 = v49;
    }
  }
  v9 = (__int64 *)(v7 + 16LL * v6);
LABEL_3:
  v56 = v9[1];
  v12 = sub_AA5930(a3);
  result = a1 + 56;
  v55 = a1 + 56;
  if ( v12 != v11 )
  {
    v14 = v11;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v12 - 8);
      v16 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) != 0 )
      {
        v17 = 0;
        do
        {
          if ( a2 == *(_QWORD *)(v15 + 32LL * *(unsigned int *)(v12 + 72) + 8 * v17) )
          {
            v16 = 32 * v17;
            goto LABEL_10;
          }
          ++v17;
        }
        while ( (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) != (_DWORD)v17 );
        v16 = 0x1FFFFFFFE0LL;
      }
LABEL_10:
      v18 = *(_QWORD *)(v15 + v16);
      v19 = *(_DWORD *)(a1 + 80);
      v20 = *(_QWORD *)(a1 + 64);
      if ( !v19 )
        goto LABEL_14;
      v21 = v19 - 1;
      v22 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
      v23 = (v19 - 1) & v22;
      v24 = (__int64 *)(v20 + 16LL * v23);
      v25 = *v24;
      if ( *v24 != v18 )
        break;
LABEL_12:
      v26 = *((_DWORD *)v24 + 2);
      v27 = 1LL << v26;
      v28 = 8LL * (v26 >> 6);
LABEL_13:
      *(_QWORD *)(*(_QWORD *)(v56 + 96) + v28) |= v27;
LABEL_14:
      result = *(_QWORD *)(v12 + 32);
      if ( !result )
        BUG();
      v12 = 0;
      if ( *(_BYTE *)(result - 24) == 84 )
        v12 = result - 24;
      if ( v14 == v12 )
        return result;
    }
    v51 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v29 = *v24;
    v50 = 1;
    do
    {
      if ( v29 == -4096 )
        goto LABEL_14;
      v30 = v50++;
      v31 = v21 & (v30 + v51);
      v51 = v31;
      v29 = *(_QWORD *)(v20 + 16 * v31);
    }
    while ( v29 != v18 );
    v52 = 1;
    v24 = (__int64 *)(v20 + 16LL * (v21 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4))));
    v32 = 0;
    while ( v25 != -4096 )
    {
      if ( !v32 && v25 == -8192 )
        v32 = v24;
      v23 = v21 & (v52 + v23);
      v24 = (__int64 *)(v20 + 16LL * v23);
      v25 = *v24;
      if ( v18 == *v24 )
        goto LABEL_12;
      ++v52;
    }
    if ( !v32 )
      v32 = v24;
    v33 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v34 = v33 + 1;
    if ( 4 * v34 >= 3 * v19 )
    {
      v53 = v14;
      sub_CE2410(v55, 2 * v19);
      v36 = *(_DWORD *)(a1 + 80);
      if ( !v36 )
        goto LABEL_63;
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 64);
      v39 = (v36 - 1) & v22;
      v14 = v53;
      v34 = *(_DWORD *)(a1 + 72) + 1;
      v32 = (__int64 *)(v38 + 16LL * v39);
      v40 = *v32;
      if ( *v32 == v18 )
        goto LABEL_29;
      v41 = 1;
      v42 = 0;
      while ( v40 != -4096 )
      {
        if ( !v42 && v40 == -8192 )
          v42 = v32;
        v39 = v37 & (v41 + v39);
        v32 = (__int64 *)(v38 + 16LL * v39);
        v40 = *v32;
        if ( v18 == *v32 )
          goto LABEL_29;
        ++v41;
      }
    }
    else
    {
      if ( v19 - *(_DWORD *)(a1 + 76) - v34 > v19 >> 3 )
        goto LABEL_29;
      v54 = v14;
      sub_CE2410(v55, v19);
      v43 = *(_DWORD *)(a1 + 80);
      if ( !v43 )
      {
LABEL_63:
        ++*(_DWORD *)(a1 + 72);
        BUG();
      }
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 64);
      v42 = 0;
      v46 = (v43 - 1) & v22;
      v14 = v54;
      v47 = 1;
      v34 = *(_DWORD *)(a1 + 72) + 1;
      v32 = (__int64 *)(v45 + 16LL * v46);
      v48 = *v32;
      if ( *v32 == v18 )
        goto LABEL_29;
      while ( v48 != -4096 )
      {
        if ( !v42 && v48 == -8192 )
          v42 = v32;
        v46 = v44 & (v47 + v46);
        v32 = (__int64 *)(v45 + 16LL * v46);
        v48 = *v32;
        if ( v18 == *v32 )
          goto LABEL_29;
        ++v47;
      }
    }
    if ( v42 )
      v32 = v42;
LABEL_29:
    *(_DWORD *)(a1 + 72) = v34;
    if ( *v32 != -4096 )
      --*(_DWORD *)(a1 + 76);
    *v32 = v18;
    v27 = 1;
    v28 = 0;
    *((_DWORD *)v32 + 2) = 0;
    goto LABEL_13;
  }
  return result;
}
