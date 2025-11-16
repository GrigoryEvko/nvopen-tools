// Function: sub_1CDE330
// Address: 0x1cde330
//
__int64 __fastcall sub_1CDE330(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // rcx
  int v4; // r10d
  __int64 v6; // rdi
  int v7; // r11d
  unsigned int v8; // edx
  unsigned int v9; // r12d
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r8
  int v14; // r13d
  __int64 v15; // r11
  __int64 *v16; // r11
  int v17; // r12d
  __int64 *v18; // r8
  int v19; // eax
  int v20; // edx
  __int64 *v21; // r11
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v23[5]; // [rsp+18h] [rbp-28h] BYREF

  v22 = a2;
  v2 = *(unsigned int *)(a1 + 32);
  if ( !(_DWORD)v2 )
    return 0xFFFFFFFFLL;
  v3 = v22;
  v4 = v2 - 1;
  v6 = *(_QWORD *)(a1 + 16);
  v7 = 1;
  v8 = (v2 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v9 = v8;
  v10 = (__int64 *)(v6 + 16LL * v8);
  v11 = *v10;
  v12 = *v10;
  if ( v22 != *v10 )
  {
    while ( 1 )
    {
      if ( v12 == -8 )
        return 0xFFFFFFFFLL;
      v14 = v7 + 1;
      v15 = v4 & (v9 + v7);
      v9 = v15;
      v16 = (__int64 *)(v6 + 16 * v15);
      v12 = *v16;
      if ( v22 == *v16 )
        break;
      v7 = v14;
    }
    if ( v16 == (__int64 *)(v6 + 16LL * (unsigned int)v2) )
      return 0xFFFFFFFFLL;
    v17 = 1;
    v18 = 0;
    while ( v11 != -8 )
    {
      if ( v11 != -16 || v18 )
        v10 = v18;
      v8 = v4 & (v17 + v8);
      v21 = (__int64 *)(v6 + 16LL * v8);
      v11 = *v21;
      if ( v22 == *v21 )
        return *((unsigned int *)v21 + 2);
      ++v17;
      v18 = v10;
      v10 = (__int64 *)(v6 + 16LL * v8);
    }
    if ( !v18 )
      v18 = v10;
    v19 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) >= (unsigned int)(3 * v2) )
    {
      LODWORD(v2) = 2 * v2;
    }
    else if ( (int)v2 - *(_DWORD *)(a1 + 28) - v20 > (unsigned int)v2 >> 3 )
    {
LABEL_16:
      *(_DWORD *)(a1 + 24) = v20;
      if ( *v18 != -8 )
        --*(_DWORD *)(a1 + 28);
      *v18 = v3;
      *((_DWORD *)v18 + 2) = 0;
      return 0;
    }
    sub_1BFE340(a1 + 8, v2);
    sub_1BFD9C0(a1 + 8, &v22, v23);
    v18 = (__int64 *)v23[0];
    v3 = v22;
    v20 = *(_DWORD *)(a1 + 24) + 1;
    goto LABEL_16;
  }
  if ( v10 != (__int64 *)(v6 + 16 * v2) )
    return *((unsigned int *)v10 + 2);
  return 0xFFFFFFFFLL;
}
