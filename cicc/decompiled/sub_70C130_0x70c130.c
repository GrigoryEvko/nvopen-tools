// Function: sub_70C130
// Address: 0x70c130
//
__int64 __fastcall sub_70C130(
        unsigned __int8 a1,
        const __m128i *a2,
        const __m128i *a3,
        _OWORD *a4,
        int *a5,
        unsigned int *a6)
{
  _OWORD *v6; // r13
  const __m128i *v8; // r10
  int v9; // r12d
  _OWORD *v10; // rcx
  const __m128i *v11; // rsi
  int v12; // r12d
  int v13; // r12d
  int v14; // r12d
  int v15; // r12d
  __int64 result; // rax
  const __m128i *v17; // [rsp+8h] [rbp-88h]
  const __m128i *v18; // [rsp+10h] [rbp-80h]
  const __m128i *v19; // [rsp+18h] [rbp-78h]
  int v23; // [rsp+48h] [rbp-48h] BYREF
  unsigned int v24; // [rsp+4Ch] [rbp-44h] BYREF
  __m128i v25[4]; // [rsp+50h] [rbp-40h] BYREF

  v6 = a4;
  sub_70BBE0(a1, a2, a3, a4, &v23, &v24);
  v8 = a3;
  *a6 = v24;
  v19 = a2 + 1;
  v9 = v23;
  v17 = v8;
  v18 = v8 + 1;
  sub_70BBE0(a1, a2 + 1, v8 + 1, v25, &v23, &v24);
  *a6 |= v24;
  v10 = v6;
  v11 = (const __m128i *)v6;
  v12 = v23 | v9;
  ++v6;
  sub_70B9E0(a1, v11, v25, v10, &v23, &v24);
  *a6 |= v24;
  v13 = v23 | v12;
  sub_70BBE0(a1, a2, v18, v6, &v23, &v24);
  *a6 |= v24;
  v14 = v23 | v13;
  sub_70BBE0(a1, v19, v17, v25, &v23, &v24);
  *a6 |= v24;
  v15 = v23 | v14;
  sub_70B8D0(a1, (const __m128i *)v6, v25, v6, &v23, &v24);
  *a5 = v23 | v15;
  result = v24;
  *a6 |= v24;
  return result;
}
