// Function: sub_203D950
// Address: 0x203d950
//
__int64 *__fastcall sub_203D950(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r13
  __int64 v10; // r10
  __int64 v11; // rcx
  unsigned __int64 v12; // r11
  unsigned int v13; // r12d
  const void **v14; // r15
  __int64 *v15; // r12
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  unsigned __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  int v21; // [rsp+28h] [rbp-38h]

  v6 = sub_20363F0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(__int64 **)(a1 + 8);
  v10 = v6;
  v11 = *(_QWORD *)(a2 + 32);
  v12 = v7;
  v13 = **(unsigned __int8 **)(a2 + 40);
  v14 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v20 = v8;
  if ( v8 )
  {
    v19 = v7;
    v17 = v11;
    v18 = v6;
    sub_1623A60((__int64)&v20, v8, 2);
    v11 = v17;
    v10 = v18;
    v12 = v19;
  }
  v21 = *(_DWORD *)(a2 + 64);
  v15 = sub_1D332F0(v9, 109, (__int64)&v20, v13, v14, 0, a3, a4, a5, v10, v12, *(_OWORD *)(v11 + 40));
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v15;
}
