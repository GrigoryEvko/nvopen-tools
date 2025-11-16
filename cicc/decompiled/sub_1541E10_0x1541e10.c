// Function: sub_1541E10
// Address: 0x1541e10
//
__int64 *__fastcall sub_1541E10(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // r13d
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // ecx
  __int64 *result; // rax
  __int64 v10; // rdx
  int v11; // r11d
  __int64 *v12; // r10
  int v13; // ecx
  int v14; // ecx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // edx
  __int64 v19; // rdi
  int v20; // r10d
  __int64 *v21; // r9
  int v22; // eax
  int v23; // edx
  __int64 v24; // rdi
  __int64 *v25; // r8
  unsigned int v26; // r14d
  int v27; // r9d
  __int64 v28; // rsi

  v4 = a1 + 472;
  v5 = *(_DWORD *)(v4 + 32);
  *(_DWORD *)(v4 + 32) = v5 + 1;
  v6 = *(_DWORD *)(a1 + 496);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 472);
    goto LABEL_14;
  }
  v7 = *(_QWORD *)(a1 + 480);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v7 + 16LL * v8);
  v10 = *result;
  if ( *result == a2 )
    goto LABEL_3;
  v11 = 1;
  v12 = 0;
  while ( v10 != -8 )
  {
    if ( !v12 && v10 == -16 )
      v12 = result;
    v8 = (v6 - 1) & (v11 + v8);
    result = (__int64 *)(v7 + 16LL * v8);
    v10 = *result;
    if ( *result == a2 )
      goto LABEL_3;
    ++v11;
  }
  v13 = *(_DWORD *)(a1 + 488);
  if ( v12 )
    result = v12;
  ++*(_QWORD *)(a1 + 472);
  v14 = v13 + 1;
  if ( 4 * v14 >= 3 * v6 )
  {
LABEL_14:
    sub_1541C50(v4, 2 * v6);
    v15 = *(_DWORD *)(a1 + 496);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 480);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 488) + 1;
      result = (__int64 *)(v17 + 16LL * v18);
      v19 = *result;
      if ( *result != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -8 )
        {
          if ( !v21 && v19 == -16 )
            v21 = result;
          v18 = v16 & (v20 + v18);
          result = (__int64 *)(v17 + 16LL * v18);
          v19 = *result;
          if ( *result == a2 )
            goto LABEL_10;
          ++v20;
        }
        if ( v21 )
          result = v21;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  if ( v6 - *(_DWORD *)(a1 + 492) - v14 <= v6 >> 3 )
  {
    sub_1541C50(v4, v6);
    v22 = *(_DWORD *)(a1 + 496);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 480);
      v25 = 0;
      v26 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = 1;
      v14 = *(_DWORD *)(a1 + 488) + 1;
      result = (__int64 *)(v24 + 16LL * v26);
      v28 = *result;
      if ( *result != a2 )
      {
        while ( v28 != -8 )
        {
          if ( !v25 && v28 == -16 )
            v25 = result;
          v26 = v23 & (v27 + v26);
          result = (__int64 *)(v24 + 16LL * v26);
          v28 = *result;
          if ( *result == a2 )
            goto LABEL_10;
          ++v27;
        }
        if ( v25 )
          result = v25;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 488);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a1 + 488) = v14;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 492);
  *result = a2;
  *((_DWORD *)result + 2) = 0;
LABEL_3:
  *((_DWORD *)result + 2) = v5;
  return result;
}
