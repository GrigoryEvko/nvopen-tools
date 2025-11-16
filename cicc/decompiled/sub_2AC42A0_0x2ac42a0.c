// Function: sub_2AC42A0
// Address: 0x2ac42a0
//
__int64 __fastcall sub_2AC42A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // esi
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 *v7; // r12
  int v8; // r10d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r13
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 384;
  v21 = a2;
  v4 = *(_DWORD *)(a1 + 408);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 384);
    v22 = 0;
LABEL_20:
    v4 *= 2;
    goto LABEL_21;
  }
  v5 = v21;
  v6 = *(_QWORD *)(a1 + 392);
  v7 = 0;
  v8 = 1;
  v9 = (v4 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v21 == *v10 )
    return v10[1];
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v7 )
      v7 = v10;
    v9 = (v4 - 1) & (v8 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v21 == *v10 )
      return v10[1];
    ++v8;
  }
  if ( !v7 )
    v7 = v10;
  v13 = *(_DWORD *)(a1 + 400);
  ++*(_QWORD *)(a1 + 384);
  v14 = v13 + 1;
  v22 = v7;
  if ( 4 * (v13 + 1) >= 3 * v4 )
    goto LABEL_20;
  if ( v4 - *(_DWORD *)(a1 + 404) - v14 <= v4 >> 3 )
  {
LABEL_21:
    sub_2AC40F0(v2, v4);
    sub_2ABE110(v2, &v21, &v22);
    v5 = v21;
    v7 = v22;
    v14 = *(_DWORD *)(a1 + 400) + 1;
  }
  *(_DWORD *)(a1 + 400) = v14;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 404);
  *v7 = v5;
  v7[1] = 0;
  v15 = sub_22077B0(0x38u);
  v20 = v15;
  if ( v15 )
    sub_2BF0340(v15, 0, v21, 0);
  sub_2AB9420(a1 + 416, v20, v16, v17, v18, v19);
  v7[1] = v20;
  return v20;
}
