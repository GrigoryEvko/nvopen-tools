// Function: sub_30B2AC0
// Address: 0x30b2ac0
//
__int64 __fastcall sub_30B2AC0(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // r8
  _QWORD *v7; // r10
  int v8; // r11d
  unsigned int v9; // eax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  int v13; // eax
  int v14; // edx
  __int64 v15; // r12
  __int64 v16; // rax
  int v17; // eax
  int v18; // ecx
  unsigned int v19; // eax
  __int64 v20; // rdi
  int v21; // r11d
  int v22; // eax
  int v23; // ecx
  int v24; // r11d
  unsigned int v25; // eax
  __int64 v26; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_19;
  }
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = 1;
  v9 = v5 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v10 = (_QWORD *)(v6 + 8LL * v9);
  v11 = *v10;
  if ( *a2 == *v10 )
    return 0;
  while ( v11 != -4096 )
  {
    if ( v7 || v11 != -8192 )
      v10 = v7;
    v9 = v5 & (v8 + v9);
    v11 = *(_QWORD *)(v6 + 8LL * v9);
    if ( *a2 == v11 )
      return 0;
    ++v8;
    v7 = v10;
    v10 = (_QWORD *)(v6 + 8LL * v9);
  }
  v13 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v10;
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v4 )
  {
LABEL_19:
    sub_30B28F0(a1, 2 * v4);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v19 = (v17 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v6 + 8LL * v19);
      v20 = *v7;
      v14 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == *a2 )
        goto LABEL_13;
      v21 = 1;
      v5 = 0;
      while ( v20 != -4096 )
      {
        if ( !v5 && v20 == -8192 )
          v5 = (__int64)v7;
        v19 = v18 & (v21 + v19);
        v7 = (_QWORD *)(v6 + 8LL * v19);
        v20 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_13;
        ++v21;
      }
LABEL_23:
      if ( v5 )
        v7 = (_QWORD *)v5;
      goto LABEL_13;
    }
LABEL_40:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v14 <= v4 >> 3 )
  {
    sub_30B28F0(a1, v4);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v5 = 0;
      v24 = 1;
      v25 = (v22 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v7 = (_QWORD *)(v6 + 8LL * v25);
      v26 = *v7;
      v14 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v7 == *a2 )
        goto LABEL_13;
      while ( v26 != -4096 )
      {
        if ( v26 == -8192 && !v5 )
          v5 = (__int64)v7;
        v25 = v23 & (v24 + v25);
        v7 = (_QWORD *)(v6 + 8LL * v25);
        v26 = *v7;
        if ( *a2 == *v7 )
          goto LABEL_13;
        ++v24;
      }
      goto LABEL_23;
    }
    goto LABEL_40;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v15 = *a2;
  *v7 = v15;
  v16 = *(unsigned int *)(a1 + 40);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v16 + 1, 8u, v6, v5);
    v16 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v16) = v15;
  ++*(_DWORD *)(a1 + 40);
  return 1;
}
