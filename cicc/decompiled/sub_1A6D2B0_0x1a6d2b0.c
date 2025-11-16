// Function: sub_1A6D2B0
// Address: 0x1a6d2b0
//
__int64 __fastcall sub_1A6D2B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // rdi
  int v6; // edx
  __int64 v7; // rsi
  int v8; // edx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r8
  int v13; // edx
  unsigned __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 *v16; // rcx
  __int64 v17; // r8
  int v18; // ecx
  int v19; // eax
  int v20; // r9d
  int v21; // r9d

  v2 = *(_QWORD *)(a1 + 224);
  v3 = *a2;
  v4 = 0;
  v5 = *(_QWORD *)(v2 + 8);
  v6 = *(_DWORD *)(v2 + 24);
  if ( (*a2 & 4) != 0 )
  {
    if ( !v6 )
      return v4;
    v7 = a2[4];
    v8 = v6 - 1;
    v9 = v8 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v10 = (__int64 *)(v5 + 16LL * v9);
    v11 = *v10;
    if ( v7 == *v10 )
      return v10[1];
    v19 = 1;
    while ( v11 != -8 )
    {
      v20 = v19 + 1;
      v9 = v8 & (v19 + v9);
      v10 = (__int64 *)(v5 + 16LL * v9);
      v11 = *v10;
      if ( v7 == *v10 )
        return v10[1];
      v19 = v20;
    }
    return 0;
  }
  if ( !v6 )
    return v4;
  v13 = v6 - 1;
  v14 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = v13 & (((unsigned int)v3 >> 4) ^ ((unsigned int)v3 >> 9));
  v16 = (__int64 *)(v5 + 16LL * v15);
  v17 = *v16;
  if ( v14 != *v16 )
  {
    v18 = 1;
    while ( v17 != -8 )
    {
      v21 = v18 + 1;
      v15 = v13 & (v18 + v15);
      v16 = (__int64 *)(v5 + 16LL * v15);
      v17 = *v16;
      if ( v14 == *v16 )
        return v16[1];
      v18 = v21;
    }
    return 0;
  }
  return v16[1];
}
