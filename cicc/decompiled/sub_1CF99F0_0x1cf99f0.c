// Function: sub_1CF99F0
// Address: 0x1cf99f0
//
__int64 __fastcall sub_1CF99F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r12
  __int64 i; // rax
  __int64 *v9; // r8
  int v10; // r10d
  __int64 v11; // rdi
  __int64 *v12; // r9
  unsigned int v13; // eax
  __int64 *v14; // rcx
  __int64 v15; // rdx
  __int64 result; // rax
  __int64 v17; // r12
  _QWORD *v18; // r14
  unsigned int v19; // esi
  int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rdx
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  int v29; // r10d
  unsigned int v30; // eax
  __int64 v31; // rsi
  int v32; // r10d

  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL);
  for ( i = *(unsigned int *)(a1 + 160); v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    if ( *(_DWORD *)(a1 + 164) <= (unsigned int)i )
    {
      sub_16CD150(a1 + 152, (const void *)(a1 + 168), 0, 8, a5, a6);
      i = *(unsigned int *)(a1 + 160);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * i) = v7;
    i = (unsigned int)(*(_DWORD *)(a1 + 160) + 1);
    *(_DWORD *)(a1 + 160) = i;
  }
  while ( (_DWORD)i )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL * (unsigned int)i - 8);
    *(_DWORD *)(a1 + 160) = i - 1;
    *(_QWORD *)(a1 + 336) = v17;
    v18 = sub_1648700(v17);
    if ( *((_BYTE *)v18 + 16) <= 0x17u )
    {
LABEL_7:
      result = *(_QWORD *)(a1 + 328);
      if ( result )
        return result;
      goto LABEL_8;
    }
    v19 = *(_DWORD *)(a1 + 320);
    if ( !v19 )
    {
      ++*(_QWORD *)(a1 + 296);
LABEL_13:
      sub_1CF9840(a1 + 296, 2 * v19);
      v20 = *(_DWORD *)(a1 + 320);
      if ( !v20 )
        goto LABEL_48;
      v17 = *(_QWORD *)(a1 + 336);
      v21 = (unsigned int)(v20 - 1);
      v22 = *(_QWORD *)(a1 + 304);
      v23 = v21 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v12 = (__int64 *)(v22 + 8LL * v23);
      v24 = *v12;
      v25 = (unsigned int)(*(_DWORD *)(a1 + 312) + 1);
      if ( *v12 != v17 )
      {
        v32 = 1;
        v9 = 0;
        while ( v24 != -8 )
        {
          if ( v24 == -16 && !v9 )
            v9 = v12;
          v23 = v21 & (v32 + v23);
          v12 = (__int64 *)(v22 + 8LL * v23);
          v24 = *v12;
          if ( v17 == *v12 )
            goto LABEL_15;
          ++v32;
        }
LABEL_39:
        if ( v9 )
          v12 = v9;
        goto LABEL_15;
      }
      goto LABEL_15;
    }
    LODWORD(v9) = v19 - 1;
    v10 = 1;
    v11 = *(_QWORD *)(a1 + 304);
    v12 = 0;
    v13 = (v19 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v14 = (__int64 *)(v11 + 8LL * v13);
    v15 = *v14;
    if ( v17 == *v14 )
      goto LABEL_7;
    while ( v15 != -8 )
    {
      if ( v15 != -16 || v12 )
        v14 = v12;
      v13 = (unsigned int)v9 & (v10 + v13);
      v15 = *(_QWORD *)(v11 + 8LL * v13);
      if ( v17 == v15 )
        goto LABEL_7;
      ++v10;
      v12 = v14;
      v14 = (__int64 *)(v11 + 8LL * v13);
    }
    v26 = *(_DWORD *)(a1 + 312);
    if ( !v12 )
      v12 = v14;
    ++*(_QWORD *)(a1 + 296);
    v25 = (unsigned int)(v26 + 1);
    if ( 4 * (int)v25 >= 3 * v19 )
      goto LABEL_13;
    v21 = v19 >> 3;
    if ( v19 - *(_DWORD *)(a1 + 316) - (unsigned int)v25 <= (unsigned int)v21 )
    {
      sub_1CF9840(a1 + 296, v19);
      v27 = *(_DWORD *)(a1 + 320);
      if ( !v27 )
      {
LABEL_48:
        ++*(_DWORD *)(a1 + 312);
        BUG();
      }
      v17 = *(_QWORD *)(a1 + 336);
      v21 = (unsigned int)(v27 - 1);
      v28 = *(_QWORD *)(a1 + 304);
      v9 = 0;
      v29 = 1;
      v30 = v21 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v12 = (__int64 *)(v28 + 8LL * v30);
      v31 = *v12;
      v25 = (unsigned int)(*(_DWORD *)(a1 + 312) + 1);
      if ( *v12 != v17 )
      {
        while ( v31 != -8 )
        {
          if ( !v9 && v31 == -16 )
            v9 = v12;
          v30 = v21 & (v29 + v30);
          v12 = (__int64 *)(v28 + 8LL * v30);
          v31 = *v12;
          if ( v17 == *v12 )
            goto LABEL_15;
          ++v29;
        }
        goto LABEL_39;
      }
    }
LABEL_15:
    *(_DWORD *)(a1 + 312) = v25;
    if ( *v12 != -8 )
      --*(_DWORD *)(a1 + 316);
    *v12 = v17;
    sub_1CF6140(a1, (__int64)v18, v25, v21, (int)v9, (int)v12);
    result = *(_QWORD *)(a1 + 328);
    if ( result )
      return result;
LABEL_8:
    LODWORD(i) = *(_DWORD *)(a1 + 160);
  }
  return 0;
}
