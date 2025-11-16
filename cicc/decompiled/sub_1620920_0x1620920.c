// Function: sub_1620920
// Address: 0x1620920
//
__int64 *__fastcall sub_1620920(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  void *v3; // r15
  __int64 v4; // rax
  size_t v5; // rdx
  void *v6; // r14
  __int64 v7; // rax
  size_t v8; // rdx
  void *v9; // r10
  __int64 v10; // rax
  size_t v11; // rdx
  void *v12; // r12
  __int64 v13; // rax
  size_t v14; // rdx
  __int64 *v15; // r13
  size_t v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rdx
  size_t v24; // [rsp+0h] [rbp-60h]
  size_t v25; // [rsp+8h] [rbp-58h]
  size_t v26; // [rsp+10h] [rbp-50h]
  void *v27; // [rsp+10h] [rbp-50h]
  void *v28; // [rsp+10h] [rbp-50h]
  size_t v29; // [rsp+18h] [rbp-48h]
  void *v30; // [rsp+28h] [rbp-38h]
  __int64 v31; // [rsp+28h] [rbp-38h]

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *(void **)(a2 + 8 * (4 - v2));
  if ( v3 )
  {
    v4 = sub_161E970(*(_QWORD *)(a2 + 8 * (4 - v2)));
    v26 = v5;
    v3 = (void *)v4;
    v2 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v26 = 0;
  }
  v6 = *(void **)(a2 + 8 * (3 - v2));
  if ( v6 )
  {
    v7 = sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v2)));
    v24 = v8;
    v6 = (void *)v7;
    v2 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v24 = 0;
  }
  v9 = *(void **)(a2 + 8 * (2 - v2));
  if ( v9 )
  {
    v10 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v2)));
    v29 = v11;
    v9 = (void *)v10;
    v2 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v29 = 0;
  }
  v12 = *(void **)(a2 + 8 * (1 - v2));
  if ( v12 )
  {
    v30 = v9;
    v13 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v2)));
    v9 = v30;
    v25 = v14;
    v12 = (void *)v13;
    v2 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v25 = 0;
  }
  v31 = *(_QWORD *)(a2 - 8 * v2);
  v15 = (__int64 *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v15 = (__int64 *)*v15;
  v16 = v26;
  v17 = 0;
  if ( v26 )
  {
    v27 = v9;
    v18 = sub_161FF10(v15, v3, v16);
    v9 = v27;
    v17 = v18;
  }
  v19 = 0;
  if ( v24 )
  {
    v28 = v9;
    v20 = sub_161FF10(v15, v6, v24);
    v9 = v28;
    v19 = v20;
  }
  v21 = 0;
  if ( v29 )
    v21 = sub_161FF10(v15, v9, v29);
  v22 = 0;
  if ( v25 )
    v22 = sub_161FF10(v15, v12, v25);
  *a1 = sub_15C1EB0(v15, v31, v22, v21, v19, v17, 2u, 1);
  return a1;
}
