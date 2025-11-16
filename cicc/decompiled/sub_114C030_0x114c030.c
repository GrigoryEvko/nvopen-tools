// Function: sub_114C030
// Address: 0x114c030
//
void __fastcall sub_114C030(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v5; // eax
  _QWORD *v6; // rdi
  _QWORD *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r8
  _QWORD *v15; // r10
  __int64 v16; // r9
  int v17; // r11d
  unsigned int v18; // edi
  _QWORD *v19; // rcx
  __int64 v20; // rdx
  int v21; // eax
  int v22; // eax
  int v23; // edx
  __int64 v24; // rsi
  _QWORD *v25; // rdi
  unsigned int v26; // r14d
  __int64 v27; // rcx
  int v28; // eax
  int v29; // ecx
  __int64 v30; // rdi
  unsigned int v31; // edx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34[8]; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return;
  v2 = a1 + 2096;
  v5 = *(_DWORD *)(a1 + 2112);
  v34[0] = a2;
  if ( !v5 )
  {
    v6 = *(_QWORD **)(a1 + 2128);
    v7 = &v6[*(unsigned int *)(a1 + 2136)];
    if ( v7 == sub_1149E10(v6, (__int64)v7, v34) )
      sub_114A990(v2, a2, v8, v9, v10, v11);
    goto LABEL_5;
  }
  v13 = *(_DWORD *)(a1 + 2120);
  if ( !v13 )
  {
    ++*(_QWORD *)(a1 + 2096);
    goto LABEL_24;
  }
  v14 = *(_QWORD *)(a1 + 2104);
  v15 = 0;
  v16 = v13 - 1;
  v17 = 1;
  v18 = v16 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = (_QWORD *)(v14 + 8LL * v18);
  v20 = *v19;
  if ( a2 != *v19 )
  {
    while ( v20 != -4096 )
    {
      if ( v15 || v20 != -8192 )
        v19 = v15;
      v18 = v16 & (v17 + v18);
      v20 = *(_QWORD *)(v14 + 8LL * v18);
      if ( a2 == v20 )
        goto LABEL_5;
      ++v17;
      v15 = v19;
      v19 = (_QWORD *)(v14 + 8LL * v18);
    }
    if ( !v15 )
      v15 = v19;
    v21 = v5 + 1;
    ++*(_QWORD *)(a1 + 2096);
    if ( 4 * v21 < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(a1 + 2116) - v21 > v13 >> 3 )
      {
LABEL_26:
        *(_DWORD *)(a1 + 2112) = v21;
        if ( *v15 != -4096 )
          --*(_DWORD *)(a1 + 2116);
        *v15 = a2;
        v33 = *(unsigned int *)(a1 + 2136);
        if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 2140) )
        {
          sub_C8D5F0(a1 + 2128, (const void *)(a1 + 2144), v33 + 1, 8u, v14, v16);
          v33 = *(unsigned int *)(a1 + 2136);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 2128) + 8 * v33) = a2;
        ++*(_DWORD *)(a1 + 2136);
        goto LABEL_5;
      }
      sub_CF4090(v2, v13);
      v22 = *(_DWORD *)(a1 + 2120);
      if ( v22 )
      {
        v23 = v22 - 1;
        v24 = *(_QWORD *)(a1 + 2104);
        v14 = 1;
        v25 = 0;
        v26 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = (_QWORD *)(v24 + 8LL * v26);
        v27 = *v15;
        v21 = *(_DWORD *)(a1 + 2112) + 1;
        if ( a2 != *v15 )
        {
          while ( v27 != -4096 )
          {
            if ( v27 == -8192 && !v25 )
              v25 = v15;
            v16 = (unsigned int)(v14 + 1);
            v26 = v23 & (v14 + v26);
            v15 = (_QWORD *)(v24 + 8LL * v26);
            v27 = *v15;
            if ( a2 == *v15 )
              goto LABEL_26;
            v14 = (unsigned int)v16;
          }
          if ( v25 )
            v15 = v25;
        }
        goto LABEL_26;
      }
LABEL_51:
      ++*(_DWORD *)(a1 + 2112);
      BUG();
    }
LABEL_24:
    sub_CF4090(v2, 2 * v13);
    v28 = *(_DWORD *)(a1 + 2120);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 2104);
      v31 = (v28 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = (_QWORD *)(v30 + 8LL * v31);
      v32 = *v15;
      v21 = *(_DWORD *)(a1 + 2112) + 1;
      if ( a2 != *v15 )
      {
        v16 = 1;
        v14 = 0;
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v14 )
            v14 = (__int64)v15;
          v31 = v29 & (v16 + v31);
          v15 = (_QWORD *)(v30 + 8LL * v31);
          v32 = *v15;
          if ( a2 == *v15 )
            goto LABEL_26;
          v16 = (unsigned int)(v16 + 1);
        }
        if ( v14 )
          v15 = (_QWORD *)v14;
      }
      goto LABEL_26;
    }
    goto LABEL_51;
  }
LABEL_5:
  v12 = *(_QWORD *)(a2 + 16);
  if ( v12 )
  {
    if ( !*(_QWORD *)(v12 + 8) )
    {
      v34[0] = *(_QWORD *)(v12 + 24);
      sub_114BD80(v2, v34);
    }
  }
}
