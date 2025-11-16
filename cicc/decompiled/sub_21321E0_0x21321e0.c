// Function: sub_21321E0
// Address: 0x21321e0
//
void __fastcall sub_21321E0(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, double a5, double a6, __m128i a7)
{
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r9
  const void ***v14; // rcx
  int v15; // edx
  __int64 v16; // r9
  __int64 *v17; // rax
  const __m128i *v18; // r9
  __int64 v19; // [rsp+0h] [rbp-60h] BYREF
  int v20; // [rsp+8h] [rbp-58h]
  _BYTE v21[8]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int8 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]

  v10 = *(_QWORD *)(a2 + 72);
  v19 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v19, v10, 2);
  v11 = a1[1];
  v12 = *a1;
  v20 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)v21,
    v12,
    *(_QWORD *)(v11 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v14 = (const void ***)sub_1D25E70(a1[1], v22, v23, v22, v23, v13, 1, 0);
  v17 = sub_1D37410(
          (__int64 *)a1[1],
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v19,
          v14,
          v15,
          v16,
          a5,
          a6,
          a7,
          *(_OWORD *)*(_QWORD *)(a2 + 32));
  *(_QWORD *)a3 = v17;
  *(_DWORD *)(a3 + 8) = 0;
  *(_QWORD *)a4 = v17;
  *(_DWORD *)(a4 + 8) = 1;
  sub_2013400((__int64)a1, a2, 1, (__int64)v17, (__m128i *)2, v18);
  if ( v19 )
    sub_161E7C0((__int64)&v19, v19);
}
