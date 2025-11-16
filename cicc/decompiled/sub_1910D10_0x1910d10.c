// Function: sub_1910D10
// Address: 0x1910d10
//
__int64 *__fastcall sub_1910D10(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // edx
  __int64 *result; // rax
  __int64 v8; // rdi
  int v9; // r11d
  __int64 *v10; // r10
  int v11; // ecx
  int v12; // ecx
  __int64 v13; // rdx
  int v14; // eax
  int v15; // esi
  __int64 v16; // r9
  unsigned int v17; // edx
  __int64 v18; // r8
  int v19; // r11d
  __int64 *v20; // r10
  _QWORD v21[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_15:
    sub_177C7D0(a1, 2 * v4);
    v14 = *(_DWORD *)(a1 + 24);
    if ( !v14 )
    {
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
    v15 = v14 - 1;
    v16 = *(_QWORD *)(a1 + 8);
    v12 = *(_DWORD *)(a1 + 16) + 1;
    v17 = (v14 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    result = (__int64 *)(v16 + 16LL * v17);
    v18 = *result;
    if ( *result != *a2 )
    {
      v19 = 1;
      v20 = 0;
      while ( v18 != -8 )
      {
        if ( !v20 && v18 == -16 )
          v20 = result;
        v17 = v15 & (v19 + v17);
        result = (__int64 *)(v16 + 16LL * v17);
        v18 = *result;
        if ( *a2 == *result )
          goto LABEL_11;
        ++v19;
      }
      if ( v20 )
        result = v20;
    }
    goto LABEL_11;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  result = (__int64 *)(v5 + 16LL * v6);
  v8 = *result;
  if ( *a2 == *result )
    return result;
  v9 = 1;
  v10 = 0;
  while ( v8 != -8 )
  {
    if ( !v10 && v8 == -16 )
      v10 = result;
    v6 = (v4 - 1) & (v9 + v6);
    result = (__int64 *)(v5 + 16LL * v6);
    v8 = *result;
    if ( *a2 == *result )
      return result;
    ++v9;
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( v10 )
    result = v10;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * v12 >= 3 * v4 )
    goto LABEL_15;
  if ( v4 - *(_DWORD *)(a1 + 20) - v12 <= v4 >> 3 )
  {
    sub_177C7D0(a1, v4);
    sub_190E590(a1, a2, v21);
    result = (__int64 *)v21[0];
    v12 = *(_DWORD *)(a1 + 16) + 1;
  }
LABEL_11:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 20);
  v13 = *a2;
  *((_DWORD *)result + 2) = 0;
  *result = v13;
  return result;
}
