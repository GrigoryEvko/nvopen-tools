// Function: sub_213C9B0
// Address: 0x213c9b0
//
__int64 *__fastcall sub_213C9B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rsi
  _QWORD *v6; // r13
  __int64 v7; // r9
  __int128 *v8; // rcx
  __int64 v9; // r10
  __int64 v10; // r11
  __int64 v11; // r15
  unsigned int v12; // r12d
  int v13; // esi
  __int64 *v14; // r12
  __int64 v16; // [rsp+0h] [rbp-60h]
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int128 *v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  int v21; // [rsp+28h] [rbp-38h]

  v3 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v5 = *(_QWORD *)(a2 + 72);
  v6 = *(_QWORD **)(a1 + 8);
  v7 = *(_QWORD *)(a2 + 104);
  v8 = *(__int128 **)(a2 + 32);
  v9 = v3;
  v10 = v4;
  v11 = *(_QWORD *)(a2 + 96);
  v12 = *(unsigned __int8 *)(a2 + 88);
  v20 = v5;
  if ( v5 )
  {
    v17 = v4;
    v18 = v7;
    v19 = v8;
    v16 = v3;
    sub_1623A60((__int64)&v20, v5, 2);
    v9 = v16;
    v10 = v17;
    v7 = v18;
    v8 = v19;
  }
  v13 = *(unsigned __int16 *)(a2 + 24);
  v21 = *(_DWORD *)(a2 + 64);
  v14 = sub_1D2B8F0(v6, v13, (__int64)&v20, v12, v11, v7, *v8, *(__int128 *)((char *)v8 + 40), v9, v10);
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v14;
}
