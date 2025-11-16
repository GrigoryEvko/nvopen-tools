// Function: sub_1620AF0
// Address: 0x1620af0
//
__int64 *__fastcall sub_1620AF0(__int64 *a1, __int64 a2)
{
  int v2; // r14d
  __int64 v3; // rax
  __int64 v4; // rbx
  void *v5; // r11
  size_t v6; // rdx
  size_t v7; // r10
  void *v8; // r8
  __int64 v9; // rax
  size_t v10; // rdx
  size_t v11; // r9
  __int64 v12; // r13
  __int64 *v13; // r15
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  size_t v19; // [rsp+0h] [rbp-60h]
  void *v20; // [rsp+0h] [rbp-60h]
  void *v21; // [rsp+8h] [rbp-58h]
  size_t v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-58h]
  int v24; // [rsp+14h] [rbp-4Ch]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h]
  char v27; // [rsp+28h] [rbp-38h]
  char v28; // [rsp+2Ch] [rbp-34h]

  v2 = *(_DWORD *)(a2 + 24);
  v24 = *(_DWORD *)(a2 + 28);
  v3 = *(unsigned int *)(a2 + 8);
  v27 = *(_BYTE *)(a2 + 32);
  v25 = *(_QWORD *)(a2 + 8 * (6 - v3));
  v28 = *(_BYTE *)(a2 + 33);
  v4 = *(_QWORD *)(a2 + 8 * (2 - v3));
  v26 = *(_QWORD *)(a2 + 8 * (3 - v3));
  v5 = *(void **)(a2 + 8 * (5 - v3));
  if ( v5 )
  {
    v5 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * (5 - v3)));
    v3 = *(unsigned int *)(a2 + 8);
    v7 = v6;
  }
  else
  {
    v7 = 0;
  }
  v8 = *(void **)(a2 + 8 * (1 - v3));
  if ( v8 )
  {
    v19 = v7;
    v21 = v5;
    v9 = sub_161E970(*(_QWORD *)(a2 + 8 * (1 - v3)));
    v5 = v21;
    v7 = v19;
    v8 = (void *)v9;
    v3 = *(unsigned int *)(a2 + 8);
    v11 = v10;
  }
  else
  {
    v11 = 0;
  }
  v12 = *(_QWORD *)(a2 - 8 * v3);
  v13 = (__int64 *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v13 = (__int64 *)*v13;
  v14 = 0;
  if ( v7 )
  {
    v20 = v8;
    v22 = v11;
    v15 = sub_161FF10(v13, v5, v7);
    v8 = v20;
    v11 = v22;
    v14 = v15;
  }
  v16 = 0;
  if ( v11 )
  {
    v23 = v14;
    v17 = sub_161FF10(v13, v8, v11);
    v14 = v23;
    v16 = v17;
  }
  *a1 = sub_15C2FB0(v13, v12, v16, v14, v4, v2, v26, v27, v28, v25, v24, 2u, 1);
  return a1;
}
