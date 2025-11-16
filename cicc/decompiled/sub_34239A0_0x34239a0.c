// Function: sub_34239A0
// Address: 0x34239a0
//
void __fastcall sub_34239A0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  const __m128i *v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rsi
  __int128 v8; // rax
  __int64 v9; // r9
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+8h] [rbp-28h]

  v4 = *(_QWORD *)(a2 + 80);
  v10 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v10, v4, 1);
  v5 = *(const __m128i **)(a1 + 64);
  v11 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  *(_QWORD *)&v8 = sub_3400BD0((__int64)v5, (__int64)v7, (__int64)&v10, 8, 0, 1u, a3, 1u);
  sub_3415CC0(v5, a2, 45, 0x106u, 0, v9, v8);
  if ( v10 )
    sub_B91220((__int64)&v10, v10);
}
