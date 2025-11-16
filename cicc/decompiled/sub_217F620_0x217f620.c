// Function: sub_217F620
// Address: 0x217f620
//
__int64 __fastcall sub_217F620(__int64 a1, int a2)
{
  unsigned int v2; // esi
  __int64 v3; // r8
  unsigned int v4; // r10d
  int v6; // r12d
  int *v7; // r9
  int v8; // eax
  unsigned int v9; // edx
  unsigned int v10; // r13d
  int *v11; // rcx
  int v12; // edi
  int v13; // r11d
  int v15; // r14d
  int *v16; // r12
  int v17; // r11d
  int *v18; // r9
  int v19; // ecx
  int v20; // ecx
  int *v21; // r12
  int v22[3]; // [rsp+Ch] [rbp-34h] BYREF
  int *v23; // [rsp+18h] [rbp-28h] BYREF

  v22[0] = a2;
  v2 = *(_DWORD *)(a1 + 32);
  if ( !v2 )
    return 0xFFFFFFFFLL;
  v3 = *(_QWORD *)(a1 + 16);
  v4 = v2 - 1;
  v6 = 1;
  v7 = (int *)(v3 + 8LL * v2);
  v8 = v22[0];
  v9 = (v2 - 1) & (37 * v22[0]);
  v10 = v9;
  v11 = (int *)(v3 + 8LL * v9);
  v12 = *v11;
  v13 = *v11;
  if ( v22[0] != *v11 )
  {
    while ( 1 )
    {
      if ( v13 == -1 )
        return 0xFFFFFFFFLL;
      v15 = v6 + 1;
      v10 = v4 & (v10 + v6);
      v16 = (int *)(v3 + 8LL * v10);
      v13 = *v16;
      if ( v22[0] == *v16 )
        break;
      v6 = v15;
    }
    if ( v7 == v16 )
      return 0xFFFFFFFFLL;
    v17 = 1;
    v18 = 0;
    while ( v12 != -1 )
    {
      if ( v12 != -2 || v18 )
        v11 = v18;
      v9 = v4 & (v17 + v9);
      v21 = (int *)(v3 + 8LL * v9);
      v12 = *v21;
      if ( v22[0] == *v21 )
        return (unsigned int)v21[1];
      ++v17;
      v18 = v11;
      v11 = (int *)(v3 + 8LL * v9);
    }
    if ( !v18 )
      v18 = v11;
    v19 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v20 = v19 + 1;
    if ( 4 * v20 >= 3 * v2 )
    {
      v2 *= 2;
    }
    else if ( v2 - *(_DWORD *)(a1 + 28) - v20 > v2 >> 3 )
    {
LABEL_16:
      *(_DWORD *)(a1 + 24) = v20;
      if ( *v18 != -1 )
        --*(_DWORD *)(a1 + 28);
      *v18 = v8;
      v18[1] = 0;
      return 0;
    }
    sub_1BFDD60(a1 + 8, v2);
    sub_1BFD720(a1 + 8, v22, &v23);
    v18 = v23;
    v8 = v22[0];
    v20 = *(_DWORD *)(a1 + 24) + 1;
    goto LABEL_16;
  }
  if ( v7 != v11 )
    return (unsigned int)v11[1];
  return 0xFFFFFFFFLL;
}
