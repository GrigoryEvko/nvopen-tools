// Function: sub_1673340
// Address: 0x1673340
//
__int64 *__fastcall sub_1673340(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 *v4; // r13
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 *result; // rax
  __int64 v10; // rdi
  int v11; // r11d
  __int64 *v12; // r10
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // rdx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // edx
  __int64 v20; // r8
  int v21; // r11d
  __int64 *v22; // r10
  int v23; // eax
  int v24; // esi
  __int64 v25; // r9
  int v26; // r11d
  unsigned int v27; // edx
  __int64 v28; // r8
  __int64 v29[8]; // [rsp+0h] [rbp-40h] BYREF

  v29[0] = *(_QWORD *)(**(_QWORD **)a1 - 8LL * *(unsigned int *)(**(_QWORD **)a1 + 8LL));
  v2 = **(_QWORD **)(a1 + 8);
  v29[2] = a2;
  v29[1] = v2;
  v3 = sub_1627350(***(__int64 ****)(a1 + 16), v29, (__int64 *)3, 0, 1);
  sub_1623BA0(**(_QWORD **)(a1 + 24), **(_DWORD **)(a1 + 32), v3);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_DWORD *)(v5 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)v5;
    goto LABEL_14;
  }
  v7 = *(_QWORD *)(v5 + 8);
  v8 = (v6 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
  result = (__int64 *)(v7 + 24LL * v8);
  v10 = *result;
  if ( *v4 == *result )
    goto LABEL_3;
  v11 = 1;
  v12 = 0;
  while ( v10 != -8 )
  {
    if ( !v12 && v10 == -16 )
      v12 = result;
    v8 = (v6 - 1) & (v11 + v8);
    result = (__int64 *)(v7 + 24LL * v8);
    v10 = *result;
    if ( *v4 == *result )
      goto LABEL_3;
    ++v11;
  }
  v13 = *(_DWORD *)(v5 + 16);
  if ( v12 )
    result = v12;
  ++*(_QWORD *)v5;
  v14 = v13 + 1;
  if ( 4 * v14 >= 3 * v6 )
  {
LABEL_14:
    sub_1672CC0(v5, 2 * v6);
    v16 = *(_DWORD *)(v5 + 24);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v5 + 8);
      v14 = *(_DWORD *)(v5 + 16) + 1;
      v19 = (v16 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      result = (__int64 *)(v18 + 24LL * v19);
      v20 = *result;
      if ( *result == *v4 )
        goto LABEL_10;
      v21 = 1;
      v22 = 0;
      while ( v20 != -8 )
      {
        if ( !v22 && v20 == -16 )
          v22 = result;
        v19 = v17 & (v21 + v19);
        result = (__int64 *)(v18 + 24LL * v19);
        v20 = *result;
        if ( *v4 == *result )
          goto LABEL_10;
        ++v21;
      }
LABEL_18:
      if ( v22 )
        result = v22;
      goto LABEL_10;
    }
LABEL_39:
    ++*(_DWORD *)(v5 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(v5 + 20) - v14 <= v6 >> 3 )
  {
    sub_1672CC0(v5, v6);
    v23 = *(_DWORD *)(v5 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v5 + 8);
      v22 = 0;
      v26 = 1;
      v14 = *(_DWORD *)(v5 + 16) + 1;
      v27 = (v23 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      result = (__int64 *)(v25 + 24LL * v27);
      v28 = *result;
      if ( *v4 == *result )
        goto LABEL_10;
      while ( v28 != -8 )
      {
        if ( !v22 && v28 == -16 )
          v22 = result;
        v27 = v24 & (v26 + v27);
        result = (__int64 *)(v25 + 24LL * v27);
        v28 = *result;
        if ( *v4 == *result )
          goto LABEL_10;
        ++v26;
      }
      goto LABEL_18;
    }
    goto LABEL_39;
  }
LABEL_10:
  *(_DWORD *)(v5 + 16) = v14;
  if ( *result != -8 )
    --*(_DWORD *)(v5 + 20);
  v15 = *v4;
  result[1] = 0;
  *((_DWORD *)result + 4) = 0;
  *result = v15;
LABEL_3:
  result[1] = v3;
  return result;
}
