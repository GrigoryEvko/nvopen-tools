// Function: sub_26D54A0
// Address: 0x26d54a0
//
__int64 __fastcall sub_26D54A0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v6; // rdx
  unsigned int v7; // esi
  __int64 v8; // rdi
  int v9; // r14d
  __int64 v10; // r9
  unsigned int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // r10
  int v15; // eax
  int v16; // ecx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rcx
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 *v25; // rdx
  __int64 v26; // [rsp+8h] [rbp-38h] BYREF
  __int64 v27; // [rsp+10h] [rbp-30h] BYREF
  int v28; // [rsp+18h] [rbp-28h]

  v6 = *a2;
  v7 = *(_DWORD *)(a1 + 24);
  v28 = 0;
  v27 = v6;
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    v26 = 0;
LABEL_22:
    v7 *= 2;
    goto LABEL_23;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v9 = 1;
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = v8 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( v6 == *(_QWORD *)v12 )
    return *(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(v12 + 8);
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = (v7 - 1) & (v9 + v11);
    v12 = v8 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( v6 == *(_QWORD *)v12 )
      return *(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(v12 + 8);
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  v26 = v10;
  if ( 4 * (v15 + 1) >= 3 * v7 )
    goto LABEL_22;
  if ( v7 - *(_DWORD *)(a1 + 20) - v16 <= v7 >> 3 )
  {
LABEL_23:
    sub_2574E40(a1, v7);
    sub_2567D60(a1, &v27, &v26);
    v6 = v27;
    v10 = v26;
    v16 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v10 = v6;
  *(_DWORD *)(v10 + 8) = v28;
  *(_DWORD *)(v10 + 8) = *(_DWORD *)(a1 + 40);
  v17 = *(unsigned int *)(a1 + 40);
  v18 = *(unsigned int *)(a1 + 44);
  v19 = *(_DWORD *)(a1 + 40);
  if ( v17 >= v18 )
  {
    v23 = *a3;
    v24 = *a2;
    if ( v18 < v17 + 1 )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v17 + 1, 0x10u, v17 + 1, v10);
      v17 = *(unsigned int *)(a1 + 40);
    }
    v25 = (__int64 *)(*(_QWORD *)(a1 + 32) + 16 * v17);
    *v25 = v24;
    v25[1] = v23;
    v20 = *(_QWORD *)(a1 + 32);
    v22 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v22;
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 32);
    v21 = (__int64 *)(v20 + 16 * v17);
    if ( v21 )
    {
      *v21 = *a2;
      v21[1] = *a3;
      v19 = *(_DWORD *)(a1 + 40);
      v20 = *(_QWORD *)(a1 + 32);
    }
    v22 = (unsigned int)(v19 + 1);
    *(_DWORD *)(a1 + 40) = v22;
  }
  return v20 + 16 * v22 - 16;
}
