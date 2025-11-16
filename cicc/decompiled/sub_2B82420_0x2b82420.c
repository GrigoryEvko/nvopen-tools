// Function: sub_2B82420
// Address: 0x2b82420
//
__int64 __fastcall sub_2B82420(__int64 a1, unsigned int *a2)
{
  int v4; // r13d
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // rdi
  _DWORD *v8; // r14
  __int64 v9; // r9
  unsigned int v10; // ecx
  _DWORD *v11; // rax
  int v12; // edx
  __int64 v13; // rax
  int v15; // eax
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rdi
  unsigned int v22; // eax
  int v23; // esi
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  unsigned int v27; // r15d
  _DWORD *v28; // rdi
  int v29; // ecx
  __int64 v30; // [rsp+4h] [rbp-3Ch]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  v10 = v6 & (37 * v4);
  v11 = (_DWORD *)(v7 + 8LL * v10);
  v12 = *v11;
  if ( v4 == *v11 )
  {
LABEL_3:
    v13 = (unsigned int)v11[1];
    return *(_QWORD *)(a1 + 32) + 12 * v13 + 4;
  }
  while ( v12 != -1 )
  {
    if ( v12 == -2 && !v8 )
      v8 = v11;
    v10 = v6 & (v9 + v10);
    v11 = (_DWORD *)(v7 + 8LL * v10);
    v12 = *v11;
    if ( v4 == *v11 )
      goto LABEL_3;
    v9 = (unsigned int)(v9 + 1);
  }
  if ( !v8 )
    v8 = v11;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v5 )
  {
LABEL_21:
    sub_A09770(a1, 2 * v5);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = (v19 - 1) & (37 * v4);
      v8 = (_DWORD *)(v21 + 8LL * v22);
      v23 = *v8;
      v16 = *(_DWORD *)(a1 + 16) + 1;
      if ( v4 != *v8 )
      {
        v9 = 1;
        v6 = 0;
        while ( v23 != -1 )
        {
          if ( !v6 && v23 == -2 )
            v6 = (__int64)v8;
          v22 = v20 & (v9 + v22);
          v8 = (_DWORD *)(v21 + 8LL * v22);
          v23 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          v9 = (unsigned int)(v9 + 1);
        }
        if ( v6 )
          v8 = (_DWORD *)v6;
      }
      goto LABEL_15;
    }
    goto LABEL_44;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
    sub_A09770(a1, v5);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v6 = 1;
      v27 = v25 & (37 * v4);
      v8 = (_DWORD *)(v26 + 8LL * v27);
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v28 = 0;
      v29 = *v8;
      if ( v4 != *v8 )
      {
        while ( v29 != -1 )
        {
          if ( !v28 && v29 == -2 )
            v28 = v8;
          v9 = (unsigned int)(v6 + 1);
          v27 = v25 & (v6 + v27);
          v8 = (_DWORD *)(v26 + 8LL * v27);
          v29 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          v6 = (unsigned int)v9;
        }
        if ( v28 )
          v8 = v28;
      }
      goto LABEL_15;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *v8 != -1 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  v8[1] = 0;
  v30 = *a2;
  v17 = *(unsigned int *)(a1 + 40);
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v17 + 1, 0xCu, v6, v9);
    v17 = *(unsigned int *)(a1 + 40);
  }
  v18 = *(_QWORD *)(a1 + 32) + 12 * v17;
  *(_QWORD *)v18 = v30;
  *(_DWORD *)(v18 + 8) = 0;
  v13 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v13 + 1;
  v8[1] = v13;
  return *(_QWORD *)(a1 + 32) + 12 * v13 + 4;
}
