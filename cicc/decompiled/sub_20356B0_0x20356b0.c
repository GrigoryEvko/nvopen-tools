// Function: sub_20356B0
// Address: 0x20356b0
//
__int64 *__fastcall sub_20356B0(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int16 *v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r12
  unsigned __int64 v10; // r10
  unsigned __int8 *v11; // rax
  __int64 v12; // rcx
  __int16 *v13; // r11
  unsigned int v14; // r15d
  const void **v15; // r13
  __int64 *v16; // r12
  unsigned __int64 v18; // [rsp+0h] [rbp-60h]
  __int16 *v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  int v22; // [rsp+28h] [rbp-38h]

  v6 = sub_2032580(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(__int64 **)(a1 + 8);
  v10 = v6;
  v11 = *(unsigned __int8 **)(a2 + 40);
  v12 = *(_QWORD *)(a2 + 32);
  v13 = v7;
  v14 = *v11;
  v15 = (const void **)*((_QWORD *)v11 + 1);
  v21 = v8;
  if ( v8 )
  {
    v19 = v7;
    v18 = v10;
    v20 = v12;
    sub_1623A60((__int64)&v21, v8, 2);
    v10 = v18;
    v13 = v19;
    v12 = v20;
  }
  v22 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D3A900(
          v9,
          0x86u,
          (__int64)&v21,
          v14,
          v15,
          0,
          a3,
          a4,
          a5,
          v10,
          v13,
          *(_OWORD *)(v12 + 40),
          *(_QWORD *)(v12 + 80),
          *(_QWORD *)(v12 + 88));
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v16;
}
