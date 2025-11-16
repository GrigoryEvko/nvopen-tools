// Function: sub_31CC7B0
// Address: 0x31cc7b0
//
__int64 __fastcall sub_31CC7B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 i; // rax
  __int64 v9; // r9
  __int64 v10; // r8
  int v11; // r11d
  _QWORD *v12; // r10
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 result; // rax
  __int64 v17; // rdx
  unsigned __int8 *v18; // r13
  unsigned int v19; // esi
  int v20; // eax
  int v21; // esi
  unsigned int v22; // eax
  __int64 v23; // rdi
  __int64 v24; // rcx
  int v25; // eax
  int v26; // eax
  int v27; // esi
  int v28; // r11d
  unsigned int v29; // eax
  __int64 v30; // rdi
  int v31; // r11d

  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL);
  for ( i = *(unsigned int *)(a1 + 168); v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    if ( i + 1 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
    {
      sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), i + 1, 8u, a5, a6);
      i = *(unsigned int *)(a1 + 168);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * i) = v7;
    i = (unsigned int)(*(_DWORD *)(a1 + 168) + 1);
    *(_DWORD *)(a1 + 168) = i;
  }
  while ( (_DWORD)i )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8LL * (unsigned int)i - 8);
    *(_DWORD *)(a1 + 168) = i - 1;
    *(_QWORD *)(a1 + 344) = v17;
    v18 = *(unsigned __int8 **)(v17 + 24);
    if ( *v18 <= 0x1Cu )
    {
LABEL_7:
      result = *(_QWORD *)(a1 + 336);
      if ( result )
        return result;
      goto LABEL_8;
    }
    v19 = *(_DWORD *)(a1 + 328);
    if ( !v19 )
    {
      ++*(_QWORD *)(a1 + 304);
LABEL_13:
      sub_313CFE0(a1 + 304, 2 * v19);
      v20 = *(_DWORD *)(a1 + 328);
      if ( !v20 )
        goto LABEL_48;
      v17 = *(_QWORD *)(a1 + 344);
      v21 = v20 - 1;
      v10 = *(_QWORD *)(a1 + 312);
      v22 = (v20 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v12 = (_QWORD *)(v10 + 8LL * v22);
      v23 = *v12;
      v24 = (unsigned int)(*(_DWORD *)(a1 + 320) + 1);
      if ( *v12 != v17 )
      {
        v31 = 1;
        v9 = 0;
        while ( v23 != -4096 )
        {
          if ( v23 == -8192 && !v9 )
            v9 = (__int64)v12;
          v22 = v21 & (v31 + v22);
          v12 = (_QWORD *)(v10 + 8LL * v22);
          v23 = *v12;
          if ( v17 == *v12 )
            goto LABEL_15;
          ++v31;
        }
LABEL_39:
        if ( v9 )
          v12 = (_QWORD *)v9;
        goto LABEL_15;
      }
      goto LABEL_15;
    }
    v9 = v19 - 1;
    v10 = *(_QWORD *)(a1 + 312);
    v11 = 1;
    v12 = 0;
    v13 = v9 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v14 = (_QWORD *)(v10 + 8LL * v13);
    v15 = *v14;
    if ( v17 == *v14 )
      goto LABEL_7;
    while ( v15 != -4096 )
    {
      if ( v15 != -8192 || v12 )
        v14 = v12;
      v13 = v9 & (v11 + v13);
      v15 = *(_QWORD *)(v10 + 8LL * v13);
      if ( v17 == v15 )
        goto LABEL_7;
      ++v11;
      v12 = v14;
      v14 = (_QWORD *)(v10 + 8LL * v13);
    }
    v25 = *(_DWORD *)(a1 + 320);
    if ( !v12 )
      v12 = v14;
    ++*(_QWORD *)(a1 + 304);
    v24 = (unsigned int)(v25 + 1);
    if ( 4 * (int)v24 >= 3 * v19 )
      goto LABEL_13;
    if ( v19 - *(_DWORD *)(a1 + 324) - (unsigned int)v24 <= v19 >> 3 )
    {
      sub_313CFE0(a1 + 304, v19);
      v26 = *(_DWORD *)(a1 + 328);
      if ( !v26 )
      {
LABEL_48:
        ++*(_DWORD *)(a1 + 320);
        BUG();
      }
      v17 = *(_QWORD *)(a1 + 344);
      v27 = v26 - 1;
      v10 = *(_QWORD *)(a1 + 312);
      v9 = 0;
      v28 = 1;
      v29 = (v26 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v12 = (_QWORD *)(v10 + 8LL * v29);
      v30 = *v12;
      v24 = (unsigned int)(*(_DWORD *)(a1 + 320) + 1);
      if ( *v12 != v17 )
      {
        while ( v30 != -4096 )
        {
          if ( !v9 && v30 == -8192 )
            v9 = (__int64)v12;
          v29 = v27 & (v28 + v29);
          v12 = (_QWORD *)(v10 + 8LL * v29);
          v30 = *v12;
          if ( v17 == *v12 )
            goto LABEL_15;
          ++v28;
        }
        goto LABEL_39;
      }
    }
LABEL_15:
    *(_DWORD *)(a1 + 320) = v24;
    if ( *v12 != -4096 )
      --*(_DWORD *)(a1 + 324);
    *v12 = v17;
    sub_31C9280(a1, v18, v17, v24, v10, v9);
    result = *(_QWORD *)(a1 + 336);
    if ( result )
      return result;
LABEL_8:
    LODWORD(i) = *(_DWORD *)(a1 + 168);
  }
  return 0;
}
