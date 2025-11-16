// Function: sub_1620150
// Address: 0x1620150
//
__int64 *__fastcall sub_1620150(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  void *v4; // r9
  size_t v5; // rdx
  size_t v6; // r10
  __int64 v7; // r13
  __int64 v8; // r15
  void *v9; // r8
  __int64 v10; // rax
  size_t v11; // rdx
  size_t v12; // r11
  int v13; // r14d
  __int64 *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  size_t v20; // [rsp+0h] [rbp-90h]
  size_t v21; // [rsp+8h] [rbp-88h]
  void *v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+8h] [rbp-88h]
  void *v24; // [rsp+10h] [rbp-80h]
  unsigned __int64 v25; // [rsp+18h] [rbp-78h]
  __int64 v26; // [rsp+20h] [rbp-70h]
  __int64 v27; // [rsp+28h] [rbp-68h]
  __int64 v28; // [rsp+30h] [rbp-60h]
  int v29; // [rsp+38h] [rbp-58h]
  unsigned int v30; // [rsp+3Ch] [rbp-54h]
  __int64 v31; // [rsp+40h] [rbp-50h]
  __int64 v32; // [rsp+48h] [rbp-48h]
  __int64 v33; // [rsp+50h] [rbp-40h]
  unsigned int v34; // [rsp+58h] [rbp-38h]
  int v35; // [rsp+5Ch] [rbp-34h]

  v3 = *(unsigned int *)(a2 + 8);
  v4 = *(void **)(a2 + 8 * (7 - v3));
  v25 = *(_QWORD *)(a2 + 8 * (8 - v3));
  if ( v4 )
  {
    v4 = (void *)sub_161E970((__int64)v4);
    v3 = *(unsigned int *)(a2 + 8);
    v6 = v5;
  }
  else
  {
    v6 = 0;
  }
  v7 = a2;
  v26 = *(_QWORD *)(a2 + 8 * (6 - v3));
  v27 = *(_QWORD *)(a2 + 8 * (5 - v3));
  v29 = *(_DWORD *)(a2 + 52);
  v28 = *(_QWORD *)(a2 + 8 * (4 - v3));
  v30 = *(_DWORD *)(a2 + 28);
  v31 = *(_QWORD *)(a2 + 40);
  v34 = *(_DWORD *)(a2 + 48);
  v32 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 8 * (1 - v3));
  v33 = *(_QWORD *)(a2 + 8 * (3 - v3));
  v35 = *(_DWORD *)(a2 + 24);
  if ( *(_BYTE *)a2 != 15 )
    v7 = *(_QWORD *)(a2 - 8 * v3);
  v9 = *(void **)(a2 + 8 * (2 - v3));
  if ( v9 )
  {
    v21 = v6;
    v24 = v4;
    v10 = sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v3)));
    v4 = v24;
    v6 = v21;
    v9 = (void *)v10;
    v12 = v11;
  }
  else
  {
    v12 = 0;
  }
  v13 = *(unsigned __int16 *)(a2 + 2);
  v14 = (__int64 *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v14 = (__int64 *)*v14;
  v15 = 0;
  if ( v6 )
  {
    v20 = v12;
    v22 = v9;
    v16 = sub_161FF10(v14, v4, v6);
    v12 = v20;
    v9 = v22;
    v15 = v16;
  }
  v17 = 0;
  if ( v12 )
  {
    v23 = v15;
    v18 = sub_161FF10(v14, v9, v12);
    v15 = v23;
    v17 = v18;
  }
  *a1 = sub_15BDB40(v14, v13, v17, v7, v35, v8, v33, v32, v34, v31, v30, v28, v29, v27, v26, v15, v25, 2u, 1);
  return a1;
}
