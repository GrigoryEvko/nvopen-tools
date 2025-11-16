// Function: sub_1FE0180
// Address: 0x1fe0180
//
__int64 __fastcall sub_1FE0180(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r10
  unsigned int v7; // esi
  size_t v8; // r8
  __int64 *v9; // r9
  int v10; // r11d
  __int64 v11; // rdi
  __int64 *v12; // r15
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 result; // rax
  int v17; // eax
  int v18; // edx
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // r10d
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  unsigned int v28; // r14d
  __int64 *v29; // rdi
  __int64 v30; // rcx
  size_t v31; // [rsp+8h] [rbp-38h]
  size_t v32; // [rsp+8h] [rbp-38h]

  v3 = a1 + 272;
  v7 = *(_DWORD *)(a1 + 296);
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 272);
    goto LABEL_18;
  }
  LODWORD(v9) = v7 - 1;
  v10 = 1;
  v11 = *(_QWORD *)(a1 + 280);
  v12 = 0;
  v13 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( a2 == *v14 )
    return *((unsigned int *)v14 + 2);
  while ( v15 != -8 )
  {
    if ( v15 == -16 && !v12 )
      v12 = v14;
    v13 = (unsigned int)v9 & (v10 + v13);
    v14 = (__int64 *)(v11 + 16LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
      return *((unsigned int *)v14 + 2);
    ++v10;
  }
  if ( !v12 )
    v12 = v14;
  v17 = *(_DWORD *)(a1 + 288);
  ++*(_QWORD *)(a1 + 272);
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v7 )
  {
LABEL_18:
    v31 = v8;
    sub_1542080(v3, 2 * v7);
    v19 = *(_DWORD *)(a1 + 296);
    if ( v19 )
    {
      v20 = v19 - 1;
      v8 = v31;
      v21 = *(_QWORD *)(a1 + 280);
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 288) + 1;
      v12 = (__int64 *)(v21 + 16LL * v22);
      v23 = *v12;
      if ( a2 != *v12 )
      {
        v24 = 1;
        v9 = 0;
        while ( v23 != -8 )
        {
          if ( !v9 && v23 == -16 )
            v9 = v12;
          v22 = v20 & (v24 + v22);
          v12 = (__int64 *)(v21 + 16LL * v22);
          v23 = *v12;
          if ( a2 == *v12 )
            goto LABEL_14;
          ++v24;
        }
        if ( v9 )
          v12 = v9;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v7 - *(_DWORD *)(a1 + 292) - v18 <= v7 >> 3 )
  {
    v32 = v8;
    sub_1542080(v3, v7);
    v25 = *(_DWORD *)(a1 + 296);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 280);
      LODWORD(v9) = 1;
      v28 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = v32;
      v18 = *(_DWORD *)(a1 + 288) + 1;
      v29 = 0;
      v12 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v12;
      if ( a2 != *v12 )
      {
        while ( v30 != -8 )
        {
          if ( !v29 && v30 == -16 )
            v29 = v12;
          v28 = v26 & ((_DWORD)v9 + v28);
          v12 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v12;
          if ( a2 == *v12 )
            goto LABEL_14;
          LODWORD(v9) = (_DWORD)v9 + 1;
        }
        if ( v29 )
          v12 = v29;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 288);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 288) = v18;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 292);
  *v12 = a2;
  *((_DWORD *)v12 + 2) = 0;
  result = sub_1E6B9A0(v8, a3, (unsigned __int8 *)byte_3F871B3, 0, v8, (int)v9);
  *((_DWORD *)v12 + 2) = result;
  return result;
}
