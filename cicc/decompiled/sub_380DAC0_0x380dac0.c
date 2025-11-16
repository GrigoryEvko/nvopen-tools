// Function: sub_380DAC0
// Address: 0x380dac0
//
unsigned __int8 *__fastcall sub_380DAC0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v8; // r15d
  __int64 v9; // r13
  int v10; // r9d
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rsi
  unsigned __int8 *v14; // r12
  __int64 v16; // rdx
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  int v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    HIWORD(v8) = 0;
    sub_2FE6CC0((__int64)&v17, *a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    LOWORD(v8) = v18;
    v9 = v19;
  }
  else
  {
    v8 = v7(*a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v9 = v16;
  }
  sub_380AAE0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *(_QWORD *)(a2 + 80);
  v12 = a1[1];
  v17 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v17, v11, 1);
  v13 = *(unsigned int *)(a2 + 24);
  v18 = *(_DWORD *)(a2 + 72);
  v14 = sub_33FAF80(v12, v13, (__int64)&v17, v8, v9, v10, a3);
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v14;
}
