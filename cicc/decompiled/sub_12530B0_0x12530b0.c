// Function: sub_12530B0
// Address: 0x12530b0
//
__int64 __fastcall sub_12530B0(__int64 a1)
{
  __int64 *v1; // r12
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  unsigned __int8 v15[52]; // [rsp+Ch] [rbp-34h] BYREF

  v1 = (__int64 *)(a1 + 112);
  v2 = sub_1253080(*(_QWORD **)(a1 + 160));
  v3 = *(_QWORD *)(a1 + 128);
  v4 = v2;
  v5 = *(_QWORD *)(a1 + 144);
  v6 = v5 - v3;
  v7 = *(_QWORD *)(a1 + 168) - (v5 - v3);
  if ( v7 )
  {
    sub_CB6C70((__int64)v1, v7);
    v5 = *(_QWORD *)(a1 + 144);
    v3 = *(_QWORD *)(a1 + 128);
  }
  if ( v3 != v5 )
    sub_CB5AE0(v1);
  ++*(_DWORD *)(a1 + 176);
  *(_WORD *)(a1 + 180) = 271;
  *(_QWORD *)(a1 + 168) = 77;
  sub_CB6C70((__int64)v1, 1u);
  *(_DWORD *)v15 = 0;
  sub_CB6200((__int64)v1, v15, 4u);
  *(_DWORD *)v15 = 0;
  sub_CB6200((__int64)v1, v15, 4u);
  sub_CB6C70((__int64)v1, 2u);
  *(_WORD *)v15 = 0;
  sub_CB6200((__int64)v1, v15, 2u);
  sub_CB6C70((__int64)v1, 0x10u);
  sub_CB6C70((__int64)v1, 0x10u);
  *(_DWORD *)v15 = (_DWORD)&loc_1000000;
  sub_CB6200((__int64)v1, v15, 4u);
  *(_WORD *)v15 = 0;
  sub_CB6200((__int64)v1, v15, 2u);
  sub_CB6C70((__int64)v1, 6u);
  v8 = *(_QWORD *)(a1 + 128);
  v9 = *(_QWORD *)(a1 + 144);
  v10 = *(_QWORD *)(a1 + 168) + v8 - v9;
  if ( v10 )
  {
    sub_CB6C70((__int64)v1, v10);
    v9 = *(_QWORD *)(a1 + 144);
    v8 = *(_QWORD *)(a1 + 128);
  }
  if ( v8 != v9 )
    sub_CB5AE0(v1);
  ++*(_DWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 168) = 77;
  *(_WORD *)(a1 + 180) = 260;
  v15[0] = 0;
  sub_CB6200((__int64)v1, v15, 1u);
  v15[0] = 0;
  sub_CB6200((__int64)v1, v15, 1u);
  sub_CB6C70((__int64)v1, 3u);
  *(_DWORD *)v15 = 0;
  sub_CB6200((__int64)v1, v15, 4u);
  *(_DWORD *)v15 = 0;
  sub_CB6200((__int64)v1, v15, 4u);
  v11 = *(_QWORD *)(a1 + 128);
  v12 = *(_QWORD *)(a1 + 144);
  v13 = *(_QWORD *)(a1 + 168) + v11 - v12;
  if ( v13 )
  {
    sub_CB6C70((__int64)v1, v13);
    v12 = *(_QWORD *)(a1 + 144);
    v11 = *(_QWORD *)(a1 + 128);
  }
  if ( v12 != v11 )
    sub_CB5AE0(v1);
  return sub_1253080(*(_QWORD **)(a1 + 160)) + *(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 128) - v6 - v4;
}
