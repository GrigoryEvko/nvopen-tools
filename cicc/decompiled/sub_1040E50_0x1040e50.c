// Function: sub_1040E50
// Address: 0x1040e50
//
__int64 *__fastcall sub_1040E50(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v6; // rcx
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r11
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // r8
  int v13; // r10d
  unsigned int v14; // edi
  __int64 *v15; // rax
  unsigned int v16; // esi
  __int64 v17; // rbx
  int v18; // esi
  int v19; // esi
  __int64 v20; // r8
  int v21; // eax
  __int64 v22; // rdi
  __int64 *result; // rax
  int v24; // eax
  int v25; // ecx
  __int64 *v26; // r8
  unsigned int v27; // r15d
  __int64 v28; // rdi
  __int64 v29; // rsi
  int v30; // edx
  int v31; // r10d
  int v32; // r15d
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 88);
  v4 = *(_QWORD *)(a1 + 72);
  if ( !(_DWORD)v3 )
    goto LABEL_62;
  v6 = ((_DWORD)v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16 * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v30 = 1;
    while ( v8 != -4096 )
    {
      v31 = v30 + 1;
      v6 = ((_DWORD)v3 - 1) & (unsigned int)(v30 + v6);
      v7 = (__int64 *)(v4 + 16LL * (unsigned int)v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v30 = v31;
    }
LABEL_62:
    BUG();
  }
LABEL_3:
  if ( v7 == (__int64 *)(v4 + 16 * v3) )
    goto LABEL_62;
  v9 = v7[1];
  v10 = 0;
  v35 = a1 + 296;
  v11 = *(_QWORD *)(v9 + 8);
  if ( v9 == v11 )
    goto LABEL_18;
  v36 = a2;
  do
  {
    while ( 1 )
    {
      v16 = *(_DWORD *)(a1 + 320);
      v17 = v11 - 32;
      if ( !v11 )
        v17 = 0;
      ++v10;
      if ( !v16 )
      {
        ++*(_QWORD *)(a1 + 296);
LABEL_12:
        v33 = v9;
        sub_1040C70(v35, 2 * v16);
        v18 = *(_DWORD *)(a1 + 320);
        if ( !v18 )
          goto LABEL_63;
        v19 = v18 - 1;
        v20 = *(_QWORD *)(a1 + 304);
        v9 = v33;
        v6 = v19 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v21 = *(_DWORD *)(a1 + 312) + 1;
        v7 = (__int64 *)(v20 + 16 * v6);
        v22 = *v7;
        if ( v17 != *v7 )
        {
          v32 = 1;
          v8 = 0;
          while ( v22 != -4096 )
          {
            if ( !v8 && v22 == -8192 )
              v8 = (__int64)v7;
            v6 = v19 & (unsigned int)(v32 + v6);
            v7 = (__int64 *)(v20 + 16LL * (unsigned int)v6);
            v22 = *v7;
            if ( v17 == *v7 )
              goto LABEL_14;
            ++v32;
          }
          if ( v8 )
            v7 = (__int64 *)v8;
        }
        goto LABEL_14;
      }
      v8 = v16 - 1;
      v12 = *(_QWORD *)(a1 + 304);
      v13 = 1;
      v7 = 0;
      v14 = v8 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v15 = (__int64 *)(v12 + 16LL * v14);
      v6 = *v15;
      if ( v17 != *v15 )
        break;
LABEL_7:
      v15[1] = v10;
      v11 = *(_QWORD *)(v11 + 8);
      if ( v9 == v11 )
        goto LABEL_17;
    }
    while ( v6 != -4096 )
    {
      if ( !v7 && v6 == -8192 )
        v7 = v15;
      v14 = v8 & (v13 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v6 = *v15;
      if ( v17 == *v15 )
        goto LABEL_7;
      ++v13;
    }
    if ( !v7 )
      v7 = v15;
    v24 = *(_DWORD *)(a1 + 312);
    ++*(_QWORD *)(a1 + 296);
    v21 = v24 + 1;
    if ( 4 * v21 >= 3 * v16 )
      goto LABEL_12;
    v6 = v16 - *(_DWORD *)(a1 + 316) - v21;
    if ( (unsigned int)v6 <= v16 >> 3 )
    {
      v34 = v9;
      sub_1040C70(v35, v16);
      v25 = *(_DWORD *)(a1 + 320);
      if ( !v25 )
      {
LABEL_63:
        ++*(_DWORD *)(a1 + 312);
        BUG();
      }
      v6 = (unsigned int)(v25 - 1);
      v26 = 0;
      v9 = v34;
      v8 = 1;
      v27 = v6 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v28 = *(_QWORD *)(a1 + 304);
      v21 = *(_DWORD *)(a1 + 312) + 1;
      v7 = (__int64 *)(v28 + 16LL * v27);
      v29 = *v7;
      if ( v17 != *v7 )
      {
        while ( v29 != -4096 )
        {
          if ( v29 == -8192 && !v26 )
            v26 = v7;
          v27 = v6 & (v8 + v27);
          v7 = (__int64 *)(v28 + 16LL * v27);
          v29 = *v7;
          if ( v17 == *v7 )
            goto LABEL_14;
          v8 = (unsigned int)(v8 + 1);
        }
        if ( v26 )
          v7 = v26;
      }
    }
LABEL_14:
    *(_DWORD *)(a1 + 312) = v21;
    if ( *v7 != -4096 )
      --*(_DWORD *)(a1 + 316);
    *v7 = v17;
    v7[1] = 0;
    v7[1] = v10;
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v9 != v11 );
LABEL_17:
  a2 = v36;
LABEL_18:
  if ( !*(_BYTE *)(a1 + 164) )
    return sub_C8CC70(a1 + 136, a2, (__int64)v7, v6, a2, v8);
  result = *(__int64 **)(a1 + 144);
  v6 = *(unsigned int *)(a1 + 156);
  v7 = &result[v6];
  if ( result == v7 )
  {
LABEL_40:
    if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 152) )
      return sub_C8CC70(a1 + 136, a2, (__int64)v7, v6, a2, v8);
    *(_DWORD *)(a1 + 156) = v6 + 1;
    *v7 = a2;
    ++*(_QWORD *)(a1 + 136);
  }
  else
  {
    while ( a2 != *result )
    {
      if ( v7 == ++result )
        goto LABEL_40;
    }
  }
  return result;
}
