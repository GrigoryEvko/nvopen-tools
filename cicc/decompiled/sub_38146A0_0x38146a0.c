// Function: sub_38146A0
// Address: 0x38146a0
//
unsigned __int8 *__fastcall sub_38146A0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r15
  __int64 v5; // r9
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 v10; // rax
  int v11; // r9d
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned __int8 *v15; // r12
  __int64 v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  int v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]

  v4 = a1[1];
  sub_375AFE0(
    a1,
    **(_QWORD **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    a3);
  v5 = *a1;
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v10 = a1[1];
  if ( v6 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v20, v5, *(_QWORD *)(v10 + 64), v8, v9);
    v12 = v22;
    v13 = (unsigned __int16)v21;
  }
  else
  {
    v13 = v6(v5, *(_QWORD *)(v10 + 64), v8, v9);
    v12 = v17;
  }
  v14 = *(_QWORD *)(a2 + 80);
  v20 = v14;
  if ( v14 )
  {
    v18 = v12;
    v19 = v13;
    sub_B96E90((__int64)&v20, v14, 1);
    v12 = v18;
    v13 = v19;
  }
  v21 = *(_DWORD *)(a2 + 72);
  v15 = sub_33FAF80(v4, 215, (__int64)&v20, v13, v12, v11, a3);
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
  return v15;
}
