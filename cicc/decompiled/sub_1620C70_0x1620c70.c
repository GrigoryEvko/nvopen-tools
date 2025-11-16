// Function: sub_1620C70
// Address: 0x1620c70
//
unsigned int **__fastcall sub_1620C70(unsigned int **a1, __int64 a2)
{
  __int64 v2; // rax
  void *v3; // r14
  __int64 v4; // rax
  size_t v5; // rdx
  void *v6; // r13
  __int64 v7; // rax
  size_t v8; // rdx
  void *v9; // r10
  size_t v10; // rdx
  size_t v11; // rbx
  __int64 *v12; // r15
  size_t v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rax
  size_t v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  size_t v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  size_t v24; // [rsp+10h] [rbp-50h]
  void *v25; // [rsp+10h] [rbp-50h]
  void *v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  int v29; // [rsp+20h] [rbp-40h]
  int v30; // [rsp+24h] [rbp-3Ch]
  __int64 v31; // [rsp+28h] [rbp-38h]

  v2 = *(unsigned int *)(a2 + 8);
  v29 = *(_DWORD *)(a2 + 28);
  v3 = *(void **)(a2 + 8 * (3 - v2));
  v28 = *(_QWORD *)(a2 + 8 * (4 - v2));
  if ( v3 )
  {
    v4 = sub_161E970((__int64)v3);
    v24 = v5;
    v3 = (void *)v4;
    v2 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v24 = 0;
  }
  v6 = *(void **)(a2 + 8 * (2 - v2));
  if ( v6 )
  {
    v7 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v2)));
    v22 = v8;
    v6 = (void *)v7;
    v2 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v22 = 0;
  }
  v9 = *(void **)(a2 - 8 * v2);
  v30 = *(_DWORD *)(a2 + 24);
  v31 = *(_QWORD *)(a2 + 8 * (1 - v2));
  if ( v9 )
  {
    v9 = (void *)sub_161E970((__int64)v9);
    v11 = v10;
  }
  else
  {
    v11 = 0;
  }
  v12 = (__int64 *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v12 = (__int64 *)*v12;
  v13 = v24;
  v14 = 0;
  if ( v24 )
  {
    v25 = v9;
    v15 = sub_161FF10(v12, v3, v13);
    v9 = v25;
    v14 = v15;
  }
  v16 = v22;
  v17 = 0;
  if ( v22 )
  {
    v23 = v14;
    v26 = v9;
    v18 = sub_161FF10(v12, v6, v16);
    v14 = v23;
    v9 = v26;
    v17 = v18;
  }
  v19 = 0;
  if ( v11 )
  {
    v27 = v14;
    v20 = sub_161FF10(v12, v9, v11);
    v14 = v27;
    v19 = v20;
  }
  *a1 = sub_15C5B60(v12, v19, v31, v30, v17, v14, v29, v28, 2u, 1);
  return a1;
}
