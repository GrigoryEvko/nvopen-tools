// Function: sub_1C0A150
// Address: 0x1c0a150
//
__int64 __fastcall sub_1C0A150(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v4; // rdi
  int v5; // r9d
  int v6; // r11d
  __int64 v7; // rcx
  unsigned int v8; // edx
  unsigned int v9; // r12d
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r10
  int v14; // r13d
  __int64 v15; // r11
  __int64 *v16; // r11
  int v17; // r13d
  __int64 *v18; // r10
  int v19; // eax
  int v20; // edx
  __int64 *v21; // r11
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v23; // [rsp+18h] [rbp-28h] BYREF

  v22 = a2;
  v2 = *(unsigned int *)(a1 + 64);
  if ( !(_DWORD)v2 )
    return 0;
  v4 = v22;
  v5 = v2 - 1;
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 48);
  v8 = (v2 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v9 = v8;
  v10 = (__int64 *)(v7 + 16LL * v8);
  v11 = *v10;
  v12 = *v10;
  if ( v22 != *v10 )
  {
    while ( 1 )
    {
      if ( v12 == -8 )
        return 0;
      v14 = v6 + 1;
      v15 = v5 & (v9 + v6);
      v9 = v15;
      v16 = (__int64 *)(v7 + 16 * v15);
      v12 = *v16;
      if ( v22 == *v16 )
        break;
      v6 = v14;
    }
    v17 = 1;
    v18 = 0;
    if ( v16 == (__int64 *)(v7 + 16LL * (unsigned int)v2) )
      return 0;
    while ( v11 != -8 )
    {
      if ( v11 != -16 || v18 )
        v10 = v18;
      v8 = v5 & (v17 + v8);
      v21 = (__int64 *)(v7 + 16LL * v8);
      v11 = *v21;
      if ( v22 == *v21 )
        return v21[1];
      ++v17;
      v18 = v10;
      v10 = (__int64 *)(v7 + 16LL * v8);
    }
    if ( !v18 )
      v18 = v10;
    v19 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) >= (unsigned int)(3 * v2) )
    {
      LODWORD(v2) = 2 * v2;
    }
    else if ( (int)v2 - *(_DWORD *)(a1 + 60) - v20 > (unsigned int)v2 >> 3 )
    {
LABEL_15:
      *(_DWORD *)(a1 + 56) = v20;
      if ( *v18 != -8 )
        --*(_DWORD *)(a1 + 60);
      *v18 = v4;
      v18[1] = 0;
      return 0;
    }
    sub_1C04E30(a1 + 40, v2);
    sub_1C09800(a1 + 40, &v22, &v23);
    v18 = v23;
    v4 = v22;
    v20 = *(_DWORD *)(a1 + 56) + 1;
    goto LABEL_15;
  }
  if ( v10 != (__int64 *)(v7 + 16 * v2) )
    return v10[1];
  return 0;
}
