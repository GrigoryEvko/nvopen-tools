// Function: sub_298C810
// Address: 0x298c810
//
__int64 *__fastcall sub_298C810(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  int v6; // r11d
  unsigned int v7; // edx
  __int64 *v8; // r8
  __int64 *v9; // rax
  __int64 v10; // rdi
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // rdx
  __int64 *v15; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v15 = 0;
LABEL_18:
    v4 *= 2;
    goto LABEL_19;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (__int64 *)(v5 + 56LL * v7);
  v9 = 0;
  v10 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  while ( v10 != -4096 )
  {
    if ( !v9 && v10 == -8192 )
      v9 = v8;
    v7 = (v4 - 1) & (v6 + v7);
    v8 = (__int64 *)(v5 + 56LL * v7);
    v10 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v6;
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v9 )
    v9 = v8;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  v15 = v9;
  if ( 4 * v13 >= 3 * v4 )
    goto LABEL_18;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_19:
    sub_298C3B0(a1, v4);
    sub_298B940(a1, a2, &v15);
    v13 = *(_DWORD *)(a1 + 16) + 1;
    v9 = v15;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  v9[6] = 0;
  *(_OWORD *)(v9 + 1) = 0;
  *v9 = v14;
  v9[5] = (__int64)(v9 + 7);
  *(_OWORD *)(v9 + 3) = 0;
  return v9 + 1;
}
