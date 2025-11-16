// Function: sub_2A10DD0
// Address: 0x2a10dd0
//
__int64 *__fastcall sub_2A10DD0(__int64 *a1, __int64 a2, char a3, char a4)
{
  unsigned __int64 v7; // r12
  int v8; // ecx
  _QWORD *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r14
  _QWORD *v12; // rax
  _QWORD *v13; // r15
  __int64 v14; // rsi
  __int64 *v15; // r8
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r12
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rcx
  int v21; // edx
  __int64 *result; // rax
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __int64 v25; // r14
  __int64 v26; // [rsp+0h] [rbp-50h]
  __int64 *v27; // [rsp+8h] [rbp-48h]
  __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == a2 + 48 )
  {
    v9 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    v8 = *(unsigned __int8 *)(v7 - 24);
    v9 = (_QWORD *)(v7 - 24);
    if ( (unsigned int)(v8 - 30) >= 0xB )
      v9 = 0;
  }
  v10 = (unsigned __int8)(a3 ^ a4);
  v26 = *(_QWORD *)((char *)v9 - 32 - 32 * v10);
  v11 = *(_QWORD *)((char *)v9 - 32 - 32LL * (unsigned int)(1 - v10));
  sub_AA5980(v11, a2, 1u);
  v12 = sub_BD2C40(72, 1u);
  v13 = v12;
  if ( v12 )
    sub_B4C8F0((__int64)v12, v26, 1u, (__int64)(v9 + 3), 0);
  v14 = v9[6];
  v15 = v13 + 6;
  v28[0] = v14;
  if ( !v14 )
  {
    if ( v15 == v28 )
      goto LABEL_11;
    v23 = v13[6];
    if ( !v23 )
      goto LABEL_11;
LABEL_16:
    v27 = v15;
    sub_B91220((__int64)v15, v23);
    v15 = v27;
    goto LABEL_17;
  }
  sub_B96E90((__int64)v28, v14, 1);
  v15 = v13 + 6;
  if ( v13 + 6 == v28 )
  {
    if ( v28[0] )
      sub_B91220((__int64)(v13 + 6), v28[0]);
    goto LABEL_11;
  }
  v23 = v13[6];
  if ( v23 )
    goto LABEL_16;
LABEL_17:
  v24 = (unsigned __int8 *)v28[0];
  v13[6] = v28[0];
  if ( v24 )
    sub_B976B0((__int64)v28, v24, (__int64)v15);
LABEL_11:
  sub_B43D60(v9);
  v18 = *a1;
  v19 = *(unsigned int *)(*a1 + 8);
  v20 = *(unsigned int *)(*a1 + 12);
  v21 = *(_DWORD *)(*a1 + 8);
  if ( v19 >= v20 )
  {
    v25 = v11 | 4;
    if ( v20 < v19 + 1 )
    {
      sub_C8D5F0(*a1, (const void *)(v18 + 16), v19 + 1, 0x10u, v16, v17);
      v19 = *(unsigned int *)(v18 + 8);
    }
    result = (__int64 *)(*(_QWORD *)v18 + 16 * v19);
    *result = a2;
    result[1] = v25;
    ++*(_DWORD *)(v18 + 8);
  }
  else
  {
    result = (__int64 *)(*(_QWORD *)v18 + 16 * v19);
    if ( result )
    {
      *result = a2;
      result[1] = v11 | 4;
      v21 = *(_DWORD *)(v18 + 8);
    }
    *(_DWORD *)(v18 + 8) = v21 + 1;
  }
  return result;
}
