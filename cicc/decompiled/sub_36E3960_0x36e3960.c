// Function: sub_36E3960
// Address: 0x36e3960
//
__int64 __fastcall sub_36E3960(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rsi
  int v8; // esi
  __int64 v9; // rax
  __int64 v10; // rdi
  int v11; // esi
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v18; // [rsp+0h] [rbp-30h] BYREF
  int v19; // [rsp+8h] [rbp-28h]

  v4 = *(_QWORD *)(a2 + 80);
  v18 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v18, v4, 1);
  v5 = *(_QWORD *)(a1 + 1136);
  v19 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = sub_36E1940(a1 + 976, (unsigned __int8)v7);
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  if ( *(_DWORD *)(v9 + 32) <= 0x40u )
    v10 = *(_QWORD *)(v9 + 24);
  else
    v10 = **(_QWORD **)(v9 + 24);
  v11 = sub_36DCF60(v10, v8, v5);
  v13 = sub_33F7740(*(_QWORD **)(a1 + 64), v11, (__int64)&v18, 1u, 0, v12, *(_OWORD *)*(_QWORD *)(a2 + 40));
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v13, v14, v15, v16);
  sub_3421DB0(v13);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return 1;
}
