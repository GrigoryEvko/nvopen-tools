// Function: sub_2AC3F50
// Address: 0x2ac3f50
//
__int64 __fastcall sub_2AC3F50(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // rdi
  int v8; // r10d
  __int64 v9; // r13
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rax
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 *v19; // rax
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  __int64 v21; // [rsp+10h] [rbp-30h] BYREF
  int v22; // [rsp+18h] [rbp-28h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v22 = 0;
  v21 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v20 = 0;
LABEL_21:
    v5 *= 2;
    goto LABEL_22;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = v6 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = v7 + 16LL * v10;
  v12 = *(_QWORD *)v11;
  if ( v4 == *(_QWORD *)v11 )
  {
LABEL_3:
    v13 = *(unsigned int *)(v11 + 8);
    return *(_QWORD *)(a1 + 32) + 16 * v13 + 8;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = v6 & (v8 + v10);
    v11 = v7 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( v4 == *(_QWORD *)v11 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  v20 = v9;
  if ( 4 * (v15 + 1) >= 3 * v5 )
    goto LABEL_21;
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
LABEL_22:
    sub_9BAAD0(a1, v5);
    sub_2ABDB00(a1, &v21, &v20);
    v4 = v21;
    v9 = v20;
    v16 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v9 = v4;
  *(_DWORD *)(v9 + 8) = v22;
  v17 = *(unsigned int *)(a1 + 40);
  v18 = *a2;
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v17 + 1, 0x10u, v6, v12);
    v17 = *(unsigned int *)(a1 + 40);
  }
  v19 = (__int64 *)(*(_QWORD *)(a1 + 32) + 16 * v17);
  *v19 = v18;
  v19[1] = 0;
  v13 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v13 + 1;
  *(_DWORD *)(v9 + 8) = v13;
  return *(_QWORD *)(a1 + 32) + 16 * v13 + 8;
}
