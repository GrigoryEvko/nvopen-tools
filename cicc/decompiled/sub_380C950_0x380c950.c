// Function: sub_380C950
// Address: 0x380c950
//
unsigned __int8 *__fastcall sub_380C950(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  _QWORD *v7; // r12
  __int128 *v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r15
  unsigned int v12; // esi
  unsigned __int8 *v13; // r12
  __int128 v15; // [rsp-10h] [rbp-60h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  int v17; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40) + 40LL * a3;
  v5 = sub_380AAE0(a1, *(_QWORD *)v4, *(_QWORD *)(v4 + 8));
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *(_QWORD **)(a1 + 8);
  v8 = *(__int128 **)(a2 + 40);
  v9 = v5;
  v11 = v10;
  v16 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v16, v6, 1);
  *((_QWORD *)&v15 + 1) = v11;
  *(_QWORD *)&v15 = v9;
  v12 = *(_DWORD *)(a2 + 24);
  v17 = *(_DWORD *)(a2 + 72);
  v13 = sub_3406EB0(v7, v12, (__int64)&v16, 1, 0, (__int64)&v16, *v8, v15);
  if ( v16 )
    sub_B91220((__int64)&v16, v16);
  return v13;
}
