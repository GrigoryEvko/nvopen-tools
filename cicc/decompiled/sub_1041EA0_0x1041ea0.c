// Function: sub_1041EA0
// Address: 0x1041ea0
//
__int64 *__fastcall sub_1041EA0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v9; // rcx
  unsigned int v10; // esi
  __int64 v11; // rbx
  __int64 v12; // rdi
  unsigned int v13; // r8d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r10
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rcx
  int v20; // r10d
  __int64 *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdx
  int v25; // eax
  int v26; // r9d
  __int64 *v27; // [rsp+8h] [rbp-48h] BYREF
  __int64 v28; // [rsp+10h] [rbp-40h] BYREF
  __int64 v29; // [rsp+18h] [rbp-38h]

  if ( *(_BYTE *)a2 != 28 )
    goto LABEL_2;
  v9 = *(_QWORD *)(a2 + 64);
  v10 = *(_DWORD *)(a1 + 56);
  v11 = a1 + 32;
  v12 = *(_QWORD *)(a1 + 40);
  if ( !v10 )
  {
    v28 = a3;
    v29 = a2;
    goto LABEL_17;
  }
  v13 = v10 - 1;
  v14 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v9 == *v15 )
  {
LABEL_5:
    *v15 = -8192;
    v10 = *(_DWORD *)(a1 + 56);
    --*(_DWORD *)(a1 + 48);
    v12 = *(_QWORD *)(a1 + 40);
    ++*(_DWORD *)(a1 + 52);
    v13 = v10 - 1;
    v28 = a3;
    v29 = a2;
    if ( v10 )
      goto LABEL_6;
LABEL_17:
    ++*(_QWORD *)(a1 + 32);
    v10 = 0;
    v27 = 0;
LABEL_18:
    v10 *= 2;
    goto LABEL_19;
  }
  v25 = 1;
  while ( v16 != -4096 )
  {
    v26 = v25 + 1;
    v14 = v13 & (v25 + v14);
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( v9 == *v15 )
      goto LABEL_5;
    v25 = v26;
  }
  v28 = a3;
  v29 = a2;
LABEL_6:
  v17 = v13 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v18 = (__int64 *)(v12 + 16LL * v17);
  v19 = *v18;
  if ( a3 == *v18 )
    goto LABEL_2;
  v20 = 1;
  v21 = 0;
  while ( v19 != -4096 )
  {
    if ( v21 || v19 != -8192 )
      v18 = v21;
    v17 = v13 & (v20 + v17);
    v19 = *(_QWORD *)(v12 + 16LL * v17);
    if ( a3 == v19 )
      goto LABEL_2;
    ++v20;
    v21 = v18;
    v18 = (__int64 *)(v12 + 16LL * v17);
  }
  if ( !v21 )
    v21 = v18;
  v22 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v23 = v22 + 1;
  v27 = v21;
  if ( 4 * v23 >= 3 * v10 )
    goto LABEL_18;
  v24 = a3;
  if ( v10 - (v23 + *(_DWORD *)(a1 + 52)) <= v10 >> 3 )
  {
LABEL_19:
    sub_103FE70(v11, v10);
    sub_103F430(v11, &v28, &v27);
    v24 = v28;
    v21 = v27;
    v23 = *(_DWORD *)(a1 + 48) + 1;
  }
  *(_DWORD *)(a1 + 48) = v23;
  if ( *v21 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v21 = v24;
  v21[1] = v29;
LABEL_2:
  sub_103D0F0(a1, a2, a3);
  return sub_1041C60(a1, a2, a3, a4);
}
