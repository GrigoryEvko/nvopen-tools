// Function: sub_2C3F200
// Address: 0x2c3f200
//
__int64 __fastcall sub_2C3F200(_QWORD **a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rdi
  int v9; // r10d
  __int64 *v10; // r13
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r9
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v23; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_2BF3F10(*a1);
  v3 = sub_2BF04D0(v2);
  if ( v3 + 112 == (*(_QWORD *)(v3 + 112) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( *(_DWORD *)(v3 + 88) != 1 )
      BUG();
    v3 = **(_QWORD **)(v3 + 80);
  }
  v4 = *(_QWORD *)(v3 + 120);
  if ( !v4 )
    BUG();
  if ( !*(_DWORD *)(v4 + 32) )
    BUG();
  v5 = (__int64)*a1;
  v6 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v4 + 24) + 40LL) + 8LL), a2, 0);
  v7 = *(_DWORD *)(v5 + 408);
  v8 = *(_QWORD *)(v5 + 392);
  v22 = v6;
  if ( !v7 )
  {
    ++*(_QWORD *)(v5 + 384);
    v23 = 0;
LABEL_27:
    v7 *= 2;
    goto LABEL_28;
  }
  v9 = 1;
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = (__int64 *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( v6 == *v12 )
    return v12[1];
  while ( v13 != -4096 )
  {
    if ( v13 == -8192 && !v10 )
      v10 = v12;
    v11 = (v7 - 1) & (v9 + v11);
    v12 = (__int64 *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( v6 == *v12 )
      return v12[1];
    ++v9;
  }
  v15 = *(_DWORD *)(v5 + 400);
  if ( !v10 )
    v10 = v12;
  ++*(_QWORD *)(v5 + 384);
  v16 = v15 + 1;
  v23 = v10;
  if ( 4 * v16 >= 3 * v7 )
    goto LABEL_27;
  if ( v7 - *(_DWORD *)(v5 + 404) - v16 <= v7 >> 3 )
  {
LABEL_28:
    sub_2AC40F0(v5 + 384, v7);
    sub_2ABE110(v5 + 384, &v22, &v23);
    v6 = v22;
    v10 = v23;
    v16 = *(_DWORD *)(v5 + 400) + 1;
  }
  *(_DWORD *)(v5 + 400) = v16;
  if ( *v10 != -4096 )
    --*(_DWORD *)(v5 + 404);
  *v10 = v6;
  v10[1] = 0;
  v17 = sub_22077B0(0x38u);
  v20 = v17;
  if ( v17 )
    sub_2BF0340(v17, 0, v22, 0, v18, v19);
  v21 = *(unsigned int *)(v5 + 424);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 428) )
  {
    sub_C8D5F0(v5 + 416, (const void *)(v5 + 432), v21 + 1, 8u, v18, v19);
    v21 = *(unsigned int *)(v5 + 424);
  }
  *(_QWORD *)(*(_QWORD *)(v5 + 416) + 8 * v21) = v20;
  ++*(_DWORD *)(v5 + 424);
  v10[1] = v20;
  return v20;
}
