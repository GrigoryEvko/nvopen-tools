// Function: sub_36DD6A0
// Address: 0x36dd6a0
//
void __fastcall sub_36DD6A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r13
  __int64 *v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int128 v13; // [rsp-10h] [rbp-50h]
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  int v17; // [rsp+18h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD **)(a1 + 64);
  v6 = *(__int64 **)(*(_QWORD *)(v3 + 40) + 40LL);
  v7 = *v6;
  v8 = v6[1];
  v16 = v4;
  if ( v4 )
  {
    v14 = v7;
    v15 = v8;
    sub_B96E90((__int64)&v16, v4, 1);
    v7 = v14;
    v8 = v15;
  }
  *((_QWORD *)&v13 + 1) = v8;
  *(_QWORD *)&v13 = v7;
  v17 = *(_DWORD *)(a2 + 72);
  v9 = sub_33F7740(v5, 7053, (__int64)&v16, 8u, 0, v8, v13);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v9, v10, v11, v12);
  sub_3421DB0(v9);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v16 )
    sub_B91220((__int64)&v16, v16);
}
