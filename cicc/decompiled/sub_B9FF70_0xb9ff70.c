// Function: sub_B9FF70
// Address: 0xb9ff70
//
__int64 __fastcall sub_B9FF70(__int64 a1, __int64 a2)
{
  int v3; // r12d
  __int64 v4; // r15
  __int64 *v5; // r14
  int v6; // r12d
  int v7; // eax
  int v8; // r10d
  unsigned int i; // ecx
  __int64 *v10; // r9
  __int64 v11; // r13
  const void *v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // r14
  __int64 v15; // r12
  __int64 result; // rax
  int v17; // eax
  unsigned int v18; // r9d
  int v19; // esi
  __int64 *v20; // rsi
  __int64 v21; // rdi
  int v22; // eax
  int v23; // r12d
  unsigned int v24; // edx
  __int64 *v25; // r8
  __int64 v26; // rcx
  int v27; // r9d
  __int64 *v28; // r10
  int v29; // eax
  __int64 *v30; // [rsp+0h] [rbp-60h]
  int v31; // [rsp+8h] [rbp-58h]
  unsigned int v32; // [rsp+Ch] [rbp-54h]
  size_t n; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v35; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 8);
  v34 = a1;
  if ( !v3 )
    goto LABEL_15;
  v5 = *(__int64 **)(a1 + 16);
  v6 = v3 - 1;
  n = *(_QWORD *)(a1 + 24) - (_QWORD)v5;
  v7 = sub_AF66D0(v5, *(_QWORD *)(a1 + 24));
  v8 = 1;
  for ( i = v6 & v7; ; i = v6 & v13 )
  {
    v10 = (__int64 *)(v4 + 8LL * i);
    v11 = *v10;
    if ( *v10 == -4096 )
    {
      v14 = *(_QWORD *)(a2 + 8);
      LODWORD(v15) = *(_DWORD *)(a2 + 24);
      goto LABEL_14;
    }
    if ( v11 != -8192 )
    {
      v12 = *(const void **)(v11 + 16);
      if ( n == *(_QWORD *)(v11 + 24) - (_QWORD)v12 )
      {
        v31 = v8;
        v32 = i;
        if ( !n )
          break;
        v30 = (__int64 *)(v4 + 8LL * i);
        v17 = memcmp(v5, v12, n);
        v10 = v30;
        i = v32;
        v8 = v31;
        if ( !v17 )
          break;
      }
    }
    v13 = i + v8++;
  }
  v14 = *(_QWORD *)(a2 + 8);
  v15 = *(unsigned int *)(a2 + 24);
  if ( v10 != (__int64 *)(v14 + 8 * v15) )
    return v11;
LABEL_14:
  if ( !(_DWORD)v15 )
  {
LABEL_15:
    ++*(_QWORD *)a2;
    v18 = 0;
    v35 = 0;
LABEL_16:
    v19 = 2 * v18;
    goto LABEL_17;
  }
  v23 = v15 - 1;
  v21 = v34;
  v24 = v23 & sub_AF66D0(*(__int64 **)(v34 + 16), *(_QWORD *)(v34 + 24));
  v25 = (__int64 *)(v14 + 8LL * v24);
  result = v34;
  v26 = *v25;
  if ( *v25 == v34 )
    return result;
  v27 = 1;
  v20 = 0;
  while ( v26 != -4096 )
  {
    if ( v26 != -8192 || v20 )
      v25 = v20;
    v24 = v23 & (v27 + v24);
    v28 = (__int64 *)(v14 + 8LL * v24);
    v26 = *v28;
    if ( *v28 == v34 )
      return result;
    ++v27;
    v20 = v25;
    v25 = (__int64 *)(v14 + 8LL * v24);
  }
  v29 = *(_DWORD *)(a2 + 16);
  v18 = *(_DWORD *)(a2 + 24);
  if ( !v20 )
    v20 = v25;
  ++*(_QWORD *)a2;
  v22 = v29 + 1;
  v35 = v20;
  if ( 4 * v22 >= 3 * v18 )
    goto LABEL_16;
  if ( v18 - (v22 + *(_DWORD *)(a2 + 20)) > v18 >> 3 )
    goto LABEL_18;
  v19 = v18;
LABEL_17:
  sub_B0CC60(a2, v19);
  sub_AFF2E0(a2, &v34, &v35);
  v20 = v35;
  v21 = v34;
  v22 = *(_DWORD *)(a2 + 16) + 1;
LABEL_18:
  *(_DWORD *)(a2 + 16) = v22;
  if ( *v20 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v20 = v21;
  return v34;
}
